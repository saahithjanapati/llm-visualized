// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createGeluDetailView } from './selectionPanelGeluPreview.js';

function createRect(width, height) {
    return {
        width,
        height,
        top: 0,
        right: width,
        bottom: height,
        left: 0,
        x: 0,
        y: 0,
        toJSON() {
            return this;
        }
    };
}

function createMockCanvasContext() {
    return {
        clearRect: () => {},
        fillRect: () => {},
        strokeRect: () => {},
        setTransform: () => {},
        beginPath: () => {},
        moveTo: () => {},
        lineTo: () => {},
        stroke: () => {},
        fill: () => {},
        arc: () => {},
        fillText: () => {},
        setLineDash: () => {}
    };
}

describe('createGeluDetailView', () => {
    beforeEach(() => {
        document.body.innerHTML = `
            <div id="detailPanel">
                <div class="detail-header"></div>
            </div>
        `;
        HTMLCanvasElement.prototype.getContext = vi.fn(() => createMockCanvasContext());
    });

    afterEach(() => {
        document.body.innerHTML = '';
        delete window.katex;
        vi.restoreAllMocks();
    });

    it('adds the Gaussian CDF relationship alongside the GPT-2 approximation', () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createGeluDetailView(panelEl);

        view.setVisible(true);

        const text = panelEl.textContent || '';
        expect(text).toContain('Exact relationship');
        expect(text).toContain('Common GPT-2 approximation');
        expect(text).toContain('CDF of a standard Gaussian');
        expect(text).toContain('standard normal distribution');
        expect(panelEl.querySelector('[data-gelu-formula]')).not.toBeNull();
        expect(panelEl.querySelector('[data-gelu-approx-formula]')).not.toBeNull();
    });

    it('shrinks the approximation formula when the detail panel is narrow', () => {
        window.katex = {
            renderToString: vi.fn((tex) => (
                `<span class="katex-display"><span class="katex"><span class="katex-html"><span class="base">${tex}</span></span></span></span>`
            ))
        };

        const panelEl = document.getElementById('detailPanel');
        const view = createGeluDetailView(panelEl);
        view.setVisible(true);

        const approxFormulaEl = panelEl.querySelector('[data-gelu-approx-formula]');
        const katexRoot = approxFormulaEl?.querySelector('.katex-display > .katex');
        const baseEl = approxFormulaEl?.querySelector('.katex-html .base');

        approxFormulaEl.getBoundingClientRect = () => createRect(132, 48);
        katexRoot.getBoundingClientRect = () => createRect(320, 26);
        baseEl.getBoundingClientRect = () => createRect(320, 26);

        view.resizeAndRender();

        const fittedFontSize = Number.parseFloat(approxFormulaEl.style.fontSize);
        expect(fittedFontSize).toBeGreaterThanOrEqual(8.25);
        expect(fittedFontSize).toBeLessThan(16);
    });
});
