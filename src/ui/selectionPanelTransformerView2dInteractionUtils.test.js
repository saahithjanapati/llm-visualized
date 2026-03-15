import { describe, expect, it } from 'vitest';

import {
    hasView2dPointerExceededClickSlop,
    resolveView2dClickSlopPx,
    resolveView2dPointerMoveIntent,
    shouldSuppressView2dDoubleClickFocus,
    shouldTreatView2dPointerReleaseAsClick
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

    it('waits for touch drags to exceed slop before panning', () => {
        expect(resolveView2dPointerMoveIntent({
            pointerType: 'touch',
            startClientX: 100,
            startClientY: 120,
            previousClientX: 100,
            previousClientY: 120,
            clientX: 101,
            clientY: 120,
            moved: false,
            suppressClick: false
        })).toEqual({
            deltaX: 1,
            deltaY: 0,
            moved: false,
            suppressClick: false,
            shouldPan: false
        });

        expect(resolveView2dPointerMoveIntent({
            pointerType: 'touch',
            startClientX: 100,
            startClientY: 120,
            previousClientX: 101,
            previousClientY: 120,
            clientX: 103,
            clientY: 120,
            moved: false,
            suppressClick: false
        })).toEqual({
            deltaX: 2,
            deltaY: 0,
            moved: true,
            suppressClick: true,
            shouldPan: true
        });
    });

    it('treats any completed pan or pinch-follow-up as non-clickable on release', () => {
        expect(shouldTreatView2dPointerReleaseAsClick({
            moved: false,
            suppressClick: false
        })).toBe(true);

        expect(shouldTreatView2dPointerReleaseAsClick({
            moved: true,
            suppressClick: false
        })).toBe(false);

        expect(shouldTreatView2dPointerReleaseAsClick({
            moved: false,
            suppressClick: true
        })).toBe(false);
    });

    it('suppresses double-click focus while a scene-backed deep detail view is active', () => {
        expect(shouldSuppressView2dDoubleClickFocus({
            headDetailDepthActive: true,
            hasActiveDetailTarget: true,
            hasDetailSceneIndex: true
        })).toBe(true);

        expect(shouldSuppressView2dDoubleClickFocus({
            headDetailDepthActive: false,
            hasActiveDetailTarget: true,
            hasDetailSceneIndex: true
        })).toBe(false);

        expect(shouldSuppressView2dDoubleClickFocus({
            headDetailDepthActive: true,
            hasActiveDetailTarget: false,
            hasDetailSceneIndex: true
        })).toBe(false);

        expect(shouldSuppressView2dDoubleClickFocus({
            headDetailDepthActive: true,
            hasActiveDetailTarget: true,
            hasDetailSceneIndex: false
        })).toBe(false);
    });
});
