import { describe, expect, it } from 'vitest';

import {
    VIEW2D_TEXT_ZOOM_BEHAVIORS,
    resolveMhsaDetailFixedTextSizing
} from './mhsaDetailFixedLabelSizing.js';

describe('mhsaDetailFixedLabelSizing', () => {
    it('uses scene-relative sizing for MHSA detail operators', () => {
        const sizing = resolveMhsaDetailFixedTextSizing({
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        });

        expect(sizing?.captionLabelScreenFontPx).toBeNull();
        expect(sizing?.captionDimensionsScreenFontPx).toBeNull();
        expect(sizing?.operatorBehavior).toBe(VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE);
        expect(sizing?.operatorMinScreenFontPx).toBeNull();
        expect(sizing?.operatorMinScreenHeightPx).toBe(0);
    });

    it('ignores non-MHSA scenes', () => {
        expect(resolveMhsaDetailFixedTextSizing({
            metadata: {
                visualContract: 'transformer-overview-v1'
            }
        })).toBeNull();
    });
});
