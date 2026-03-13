import { describe, expect, it } from 'vitest';

import { resolveMhsaDimensionVisualExtent } from '../src/view2d/shared/mhsaDimensionSizing.js';

describe('mhsaDimensionSizing', () => {
    it('keeps larger MHSA dimensions visually larger than smaller ones', () => {
        const one = resolveMhsaDimensionVisualExtent(1);
        const twelve = resolveMhsaDimensionVisualExtent(12);
        const twentyFive = resolveMhsaDimensionVisualExtent(25);
        const sixtyFour = resolveMhsaDimensionVisualExtent(64);
        const sevenSixtyEight = resolveMhsaDimensionVisualExtent(768);

        expect(one).toBeLessThan(twelve);
        expect(twelve).toBeLessThan(twentyFive);
        expect(twentyFive).toBeLessThan(sixtyFour);
        expect(sixtyFour).toBeLessThan(sevenSixtyEight);
        expect(sevenSixtyEight - sixtyFour).toBeGreaterThanOrEqual(40);
    });
});
