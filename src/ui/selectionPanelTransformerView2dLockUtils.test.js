import { describe, expect, it } from 'vitest';

import {
    hasTransformerView2dLockedDetailSelection,
    isTransformerView2dDetailSelectionLockActive,
    resolveTransformerView2dDetailClickLockAction,
    shouldFreezeTransformerView2dDetailHover,
    TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS
} from './selectionPanelTransformerView2dLockUtils.js';

describe('selectionPanelTransformerView2dLockUtils', () => {
    it('detects when a detail-scene focus is locked', () => {
        expect(hasTransformerView2dLockedDetailSelection(null)).toBe(false);
        expect(hasTransformerView2dLockedDetailSelection(undefined)).toBe(false);
        expect(hasTransformerView2dLockedDetailSelection({
            activeNodeIds: ['node-a']
        })).toBe(true);
    });

    it('freezes detail hover only when hover is allowed and the detail focus is locked', () => {
        expect(shouldFreezeTransformerView2dDetailHover({
            allowDetailSceneHover: false,
            detailSceneSelectionLocked: true
        })).toBe(false);

        expect(shouldFreezeTransformerView2dDetailHover({
            allowDetailSceneHover: true,
            detailSceneSelectionLocked: false
        })).toBe(false);

        expect(shouldFreezeTransformerView2dDetailHover({
            allowDetailSceneHover: true,
            detailSceneSelectionLocked: true
        })).toBe(true);
    });

    it('treats a pinned-but-not-click-locked focus as hoverable', () => {
        expect(hasTransformerView2dLockedDetailSelection({
            activeNodeIds: ['node-a']
        })).toBe(true);
        expect(isTransformerView2dDetailSelectionLockActive(false)).toBe(false);
        expect(shouldFreezeTransformerView2dDetailHover({
            allowDetailSceneHover: true,
            detailSceneSelectionLocked: false
        })).toBe(false);
    });

    it('keeps a locked detail selection when clicking another focusable detail target', () => {
        expect(resolveTransformerView2dDetailClickLockAction({
            detailSceneSelectionLocked: true,
            detailHoverState: {
                focusState: {
                    activeNodeIds: ['node-b']
                }
            }
        })).toBe(TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.IGNORE);
    });

    it('clears a locked detail selection only when clicking blank canvas', () => {
        expect(resolveTransformerView2dDetailClickLockAction({
            detailSceneSelectionLocked: true,
            detailHoverState: null
        })).toBe(TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.CLEAR_LOCK);
    });

    it('locks an unlocked detail selection when clicking a focusable detail target', () => {
        expect(resolveTransformerView2dDetailClickLockAction({
            detailSceneSelectionLocked: false,
            detailHoverState: {
                focusState: {
                    activeNodeIds: ['node-a']
                }
            }
        })).toBe(TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.LOCK_TARGET);
    });

    it('does nothing for unlocked blank-canvas clicks', () => {
        expect(resolveTransformerView2dDetailClickLockAction({
            detailSceneSelectionLocked: false,
            detailHoverState: null
        })).toBe(TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.NONE);
    });
});
