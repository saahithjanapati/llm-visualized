import { describe, expect, it } from 'vitest';

import {
    hasView2dPointerExceededClickSlop,
    resolveView2dClickSlopPx
} from './selectionPanelTransformerView2dInteractionUtils.js';

describe('selectionPanelTransformerView2dInteractionUtils', () => {
    it('uses a tighter click slop for touch pointers', () => {
        expect(resolveView2dClickSlopPx('mouse')).toBe(6);
        expect(resolveView2dClickSlopPx('touch')).toBe(2);
    });

    it('treats short touch drags as pans before desktop would', () => {
        expect(hasView2dPointerExceededClickSlop({
            pointerType: 'touch',
            startClientX: 100,
            startClientY: 120,
            clientX: 103,
            clientY: 120
        })).toBe(true);

        expect(hasView2dPointerExceededClickSlop({
            pointerType: 'mouse',
            startClientX: 100,
            startClientY: 120,
            clientX: 103,
            clientY: 120
        })).toBe(false);
    });

    it('still allows stationary touch taps to behave like clicks', () => {
        expect(hasView2dPointerExceededClickSlop({
            pointerType: 'touch',
            startClientX: 64,
            startClientY: 48,
            clientX: 64,
            clientY: 48
        })).toBe(false);
    });
});
