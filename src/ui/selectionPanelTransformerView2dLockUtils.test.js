import { describe, expect, it } from 'vitest';

import {
    hasTransformerView2dLockedDetailSelection,
    shouldFreezeTransformerView2dDetailHover
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
            detailScenePinnedFocus: {
                activeNodeIds: ['node-a']
            }
        })).toBe(false);

        expect(shouldFreezeTransformerView2dDetailHover({
            allowDetailSceneHover: true,
            detailScenePinnedFocus: null
        })).toBe(false);

        expect(shouldFreezeTransformerView2dDetailHover({
            allowDetailSceneHover: true,
            detailScenePinnedFocus: {
                activeNodeIds: ['node-a']
            }
        })).toBe(true);
    });
});
