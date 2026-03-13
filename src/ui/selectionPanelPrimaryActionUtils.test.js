import { describe, expect, it } from 'vitest';

import { MHSA_INFO_PANEL_ACTION_OPEN } from './selectionPanelMhsaAction.js';
import { resolveSelectionPrimaryActionConfig } from './selectionPanelPrimaryActionUtils.js';
import { TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN } from './selectionPanelTransformerView2d.js';

describe('resolveSelectionPrimaryActionConfig', () => {
    it('routes MHSA head focus actions to the transformer 2D canvas instead of the CSS inspector', () => {
        const config = resolveSelectionPrimaryActionConfig({
            view2dContext: {
                focusLabel: 'Layer 3 Attention Head 6'
            }
        });

        expect(config).toEqual({
            action: TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN,
            label: 'View in 2D / matrix form',
            ariaLabel: 'View Layer 3 Attention Head 6 in 2D / matrix form',
            title: 'View Layer 3 Attention Head 6 in 2D / matrix form'
        });
        expect(config.action).not.toBe(MHSA_INFO_PANEL_ACTION_OPEN);
    });

    it('disables the primary action when there is no 2D focus context', () => {
        expect(resolveSelectionPrimaryActionConfig()).toBeNull();
    });
});
