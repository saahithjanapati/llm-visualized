import { describe, expect, it } from 'vitest';
import {
    resolveMhsaTokenMatrixFixedLabelScale
} from '../src/ui/selectionPanelMhsaViewportUtils.js';

describe('selectionPanelMhsaViewportUtils', () => {
    it('keeps labels at their base size when the viewport scale is neutral', () => {
        expect(resolveMhsaTokenMatrixFixedLabelScale(1)).toBe(1);
    });

    it('counter-scales labels when the DOM graph is zoomed out', () => {
        expect(resolveMhsaTokenMatrixFixedLabelScale(0.5)).toBe(2);
    });

    it('counter-scales labels when the DOM graph is zoomed in', () => {
        expect(resolveMhsaTokenMatrixFixedLabelScale(2)).toBe(0.5);
    });

    it('falls back to neutral scaling for invalid viewport values', () => {
        expect(resolveMhsaTokenMatrixFixedLabelScale(0)).toBe(1);
        expect(resolveMhsaTokenMatrixFixedLabelScale(Number.NaN)).toBe(1);
        expect(resolveMhsaTokenMatrixFixedLabelScale(-3)).toBe(1);
    });
});
