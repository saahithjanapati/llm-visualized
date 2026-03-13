import { describe, expect, it } from 'vitest';

import { resolveMhsaTokenMatrixLayoutMetrics } from './selectionPanelMhsaLayoutUtils.js';

describe('resolveMhsaTokenMatrixLayoutMetrics', () => {
    it('keeps 12-token attention grids readable while preserving denser scaling at 25 tokens', () => {
        const mediumMetrics = resolveMhsaTokenMatrixLayoutMetrics({
            rowCount: 12
        });
        const denseMetrics = resolveMhsaTokenMatrixLayoutMetrics({
            rowCount: 25
        });

        expect(mediumMetrics.componentOverrides.gridCellSize).toBeGreaterThanOrEqual(5);
        expect(denseMetrics.componentOverrides.gridCellSize).toBeGreaterThanOrEqual(3);
        expect(denseMetrics.componentOverrides.gridCellSize).toBeLessThanOrEqual(
            mediumMetrics.componentOverrides.gridCellSize
        );
        expect(denseMetrics.componentOverrides.gridCellGap).toBeLessThanOrEqual(
            mediumMetrics.componentOverrides.gridCellGap
        );
    });
});
