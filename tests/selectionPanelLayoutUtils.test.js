import { describe, expect, it } from 'vitest';
import {
    clampDesktopSelectionPanelWidth,
    resolveDesktopSelectionPanelWidthBounds
} from '../src/ui/selectionPanelLayoutUtils.js';

describe('selectionPanelLayoutUtils', () => {
    it('uses the configured desktop resize bounds on wide viewports', () => {
        expect(resolveDesktopSelectionPanelWidthBounds({ viewportWidth: 1440 })).toEqual({
            minWidthPx: 320,
            maxWidthPx: 760
        });
    });

    it('clamps resized widths into the desktop bounds', () => {
        expect(clampDesktopSelectionPanelWidth(900, { viewportWidth: 1440 })).toBe(760);
        expect(clampDesktopSelectionPanelWidth(280, { viewportWidth: 1440 })).toBe(320);
        expect(clampDesktopSelectionPanelWidth(520, { viewportWidth: 1440 })).toBe(520);
    });

    it('shrinks the allowed range when the viewport cannot fit the default minimum', () => {
        expect(resolveDesktopSelectionPanelWidthBounds({ viewportWidth: 420 })).toEqual({
            minWidthPx: 300,
            maxWidthPx: 300
        });
        expect(clampDesktopSelectionPanelWidth(520, { viewportWidth: 420 })).toBe(300);
    });
});
