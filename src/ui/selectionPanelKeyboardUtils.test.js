import { describe, expect, it } from 'vitest';

import {
    hasSelectionPanelKeyboardOwnership,
    shouldCaptureMhsaKeyboardInput
} from './selectionPanelKeyboardUtils.js';

function createPanel(activeChild = null) {
    return {
        contains(node) {
            return node === activeChild;
        }
    };
}

describe('selectionPanelKeyboardUtils', () => {
    it('treats an explicitly engaged panel as the keyboard owner', () => {
        expect(hasSelectionPanelKeyboardOwnership({
            panel: createPanel(),
            keyboardEngaged: true,
            activeElement: null
        })).toBe(true);
    });

    it('treats a focused panel element as the keyboard owner', () => {
        const child = {};
        expect(hasSelectionPanelKeyboardOwnership({
            panel: createPanel(child),
            keyboardEngaged: false,
            activeElement: child
        })).toBe(true);
    });

    it('does not treat a background panel as the keyboard owner', () => {
        expect(hasSelectionPanelKeyboardOwnership({
            panel: createPanel({}),
            keyboardEngaged: false,
            activeElement: {}
        })).toBe(false);
    });

    it('only captures MHSA keyboard controls when the MHSA panel is both visible and active', () => {
        const child = {};
        expect(shouldCaptureMhsaKeyboardInput({
            isOpen: true,
            isMhsaInfoSelectionActive: true,
            mhsaTokenMatrixHidden: false,
            panel: createPanel(child),
            keyboardEngaged: false,
            activeElement: child
        })).toBe(true);

        expect(shouldCaptureMhsaKeyboardInput({
            isOpen: true,
            isMhsaInfoSelectionActive: true,
            mhsaTokenMatrixHidden: false,
            panel: createPanel(child),
            keyboardEngaged: false,
            activeElement: {}
        })).toBe(false);
    });
});
