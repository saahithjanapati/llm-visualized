import { describe, expect, it } from 'vitest';
import {
    resolveMhsaTokenMatrixCanvasMode,
    shouldRenderMhsaTokenMatrixCanvas,
    VIEW2D_MHSA_CANVAS_MODES
} from '../src/view2d/runtime/view2dFeatureFlags.js';

describe('view2dFeatureFlags', () => {
    it('defaults the MHSA canvas mode to off', () => {
        expect(resolveMhsaTokenMatrixCanvasMode({})).toBe(VIEW2D_MHSA_CANVAS_MODES.OFF);
        expect(shouldRenderMhsaTokenMatrixCanvas(VIEW2D_MHSA_CANVAS_MODES.OFF)).toBe(false);
    });

    it('accepts replace-oriented truthy aliases for the MHSA canvas mode', () => {
        expect(resolveMhsaTokenMatrixCanvasMode({ __MHSA_TOKEN_MATRIX_CANVAS_MODE: true }))
            .toBe(VIEW2D_MHSA_CANVAS_MODES.REPLACE);
        expect(resolveMhsaTokenMatrixCanvasMode({ __MHSA_TOKEN_MATRIX_CANVAS_MODE: 'replace' }))
            .toBe(VIEW2D_MHSA_CANVAS_MODES.REPLACE);
        expect(resolveMhsaTokenMatrixCanvasMode({ __MHSA_TOKEN_MATRIX_CANVAS_MODE: 'canvas' }))
            .toBe(VIEW2D_MHSA_CANVAS_MODES.REPLACE);
        expect(shouldRenderMhsaTokenMatrixCanvas(VIEW2D_MHSA_CANVAS_MODES.REPLACE)).toBe(true);
    });

    it('accepts explicit off values and treats unsupported values as off', () => {
        expect(resolveMhsaTokenMatrixCanvasMode({ __MHSA_TOKEN_MATRIX_CANVAS_MODE: false }))
            .toBe(VIEW2D_MHSA_CANVAS_MODES.OFF);
        expect(resolveMhsaTokenMatrixCanvasMode({ __MHSA_TOKEN_MATRIX_CANVAS_MODE: 'off' }))
            .toBe(VIEW2D_MHSA_CANVAS_MODES.OFF);
        expect(resolveMhsaTokenMatrixCanvasMode({ __MHSA_TOKEN_MATRIX_CANVAS_MODE: 'false' }))
            .toBe(VIEW2D_MHSA_CANVAS_MODES.OFF);
        expect(resolveMhsaTokenMatrixCanvasMode({ __MHSA_TOKEN_MATRIX_CANVAS_MODE: 'overlay' }))
            .toBe(VIEW2D_MHSA_CANVAS_MODES.OFF);
        expect(resolveMhsaTokenMatrixCanvasMode({ __MHSA_TOKEN_MATRIX_CANVAS_MODE: 'nope' }))
            .toBe(VIEW2D_MHSA_CANVAS_MODES.OFF);
    });
});
