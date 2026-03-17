import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

const eagerTowerCss = readFileSync(
    new URL('../../../public/twelve-layer-stack.css', import.meta.url),
    'utf8'
);
const lazySelectionPanelCss = readFileSync(
    new URL('../../ui/selectionPanel.css', import.meta.url),
    'utf8'
);

describe('startup hover label styles', () => {
    it('keeps 3D hover token-chip selectors in the eager tower stylesheet', () => {
        const sharedSelectors = [
            '.detail-subtitle-token-chip',
            '.scene-hover-label__content',
            '.scene-hover-label__top-row',
            '.scene-hover-label__token-chip',
            '.scene-hover-label__attention-details',
            '.scene-hover-label__attention-chip',
            '.scene-hover-label__attention-metric-value'
        ];

        sharedSelectors.forEach((selector) => {
            expect(eagerTowerCss).toContain(selector);
            expect(lazySelectionPanelCss).not.toContain(selector);
        });
    });
});
