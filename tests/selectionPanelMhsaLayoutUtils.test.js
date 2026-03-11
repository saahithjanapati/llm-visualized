import { describe, expect, it } from 'vitest';
import {
    resolveMhsaTokenMatrixLayoutMetrics
} from '../src/ui/selectionPanelMhsaLayoutUtils.js';

describe('selectionPanelMhsaLayoutUtils', () => {
    it('keeps spacing boosts at zero for the baseline five-token layout', () => {
        const metrics = resolveMhsaTokenMatrixLayoutMetrics({ rowCount: 5, isSmallScreen: false });

        expect(metrics.extraRows).toBe(0);
        expect(metrics.cssVars['--mhsa-token-matrix-stage-gap-boost']).toBe('0px');
        expect(metrics.cssVars['--mhsa-token-matrix-attention-flow-gap-boost']).toBe('0px');
        expect(metrics.cssVars['--mhsa-token-matrix-head-copy-offset-boost']).toBe('0px');
        expect(metrics.connectorGaps.default).toBe(10);
        expect(metrics.connectorGaps.post).toBe(8);
        expect(metrics.connectorGaps.value).toBe(18);
    });

    it('increases horizontal spacing and connector gaps as token matrices grow', () => {
        const metrics = resolveMhsaTokenMatrixLayoutMetrics({ rowCount: 8, isSmallScreen: false });

        expect(metrics.extraRows).toBe(3);
        expect(metrics.cssVars['--mhsa-token-matrix-stack-column-gap-boost']).toBe('48px');
        expect(metrics.cssVars['--mhsa-token-matrix-attention-flow-gap-boost']).toBe('30px');
        expect(metrics.cssVars['--mhsa-token-matrix-head-copy-offset-boost']).toBe('48px');
        expect(metrics.connectorGaps.default).toBe(16);
        expect(metrics.connectorGaps.post).toBe(14);
        expect(metrics.connectorGaps.value).toBe(27);
    });

    it('uses gentler spacing growth on small screens', () => {
        const desktop = resolveMhsaTokenMatrixLayoutMetrics({ rowCount: 9, isSmallScreen: false });
        const mobile = resolveMhsaTokenMatrixLayoutMetrics({ rowCount: 9, isSmallScreen: true });

        expect(Number.parseInt(mobile.cssVars['--mhsa-token-matrix-attention-flow-gap-boost'], 10))
            .toBeLessThan(Number.parseInt(desktop.cssVars['--mhsa-token-matrix-attention-flow-gap-boost'], 10));
        expect(mobile.connectorGaps.value).toBeLessThan(desktop.connectorGaps.value);
    });
});
