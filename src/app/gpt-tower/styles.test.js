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

describe('startup mobile chrome styles', () => {
    it('keeps top-control mobile sizing rules in the eager tower stylesheet', () => {
        const eagerSelectors = [
            '#topControls > * {',
            '#topControls[data-auto-hidden="true"] > * {',
            'width: clamp(40px, 11.5vw, 42px);',
            'width: clamp(66px, 21vw, 84px);',
            'width: clamp(50px, 16vw, 66px);'
        ];

        eagerSelectors.forEach((selector) => {
            expect(eagerTowerCss).toContain(selector);
            expect(lazySelectionPanelCss).not.toContain(selector);
        });
    });
});
