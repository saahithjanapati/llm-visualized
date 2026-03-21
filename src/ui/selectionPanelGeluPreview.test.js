// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
    createGeluDetailView,
    isGeluDetailSelection
} from './selectionPanelGeluPreview.js';

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

    it('uses the standardized GELU cards without a separate exact-relationship block', () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createGeluDetailView(panelEl);

        view.setVisible(true);

        const text = panelEl.textContent || '';
        expect(text).not.toContain('Exact relationship');
        expect(text).toContain('What It Does');
        expect(text).toContain('Why GPT-2 Uses It');
        expect(text).toContain('Interactive curve');
        expect(panelEl.querySelector('[data-gelu-approx-formula]')).not.toBeNull();
        expect(panelEl.querySelector('[data-gelu-inline="canonical"]')).not.toBeNull();
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

    it('renders inline GELU math with KaTeX instead of plain fallback text', () => {
        window.katex = {
            renderToString: vi.fn((tex, options = {}) => (
                `<span data-display-mode="${options.displayMode ? 'true' : 'false'}">${tex}</span>`
            ))
        };

        const panelEl = document.getElementById('detailPanel');
        const view = createGeluDetailView(panelEl);

        view.setVisible(true);

        expect(window.katex.renderToString).toHaveBeenCalledWith(
            '\\operatorname{GELU}(x)',
            expect.objectContaining({ displayMode: false, throwOnError: false })
        );
        const inlineFormula = panelEl.querySelector('[data-gelu-inline="geluOfX"]');
        expect(inlineFormula?.innerHTML).toContain('data-display-mode="false"');
    });

    it('treats the post-GELU activation vector as eligible for the GELU detail action', () => {
        expect(isGeluDetailSelection({
            label: 'MLP Activation (post GELU)',
            stage: 'mlp.activation'
        })).toBe(true);
        expect(isGeluDetailSelection({
            label: 'MLP Up Weight Matrix',
            stage: 'mlp.up.weight'
        })).toBe(true);
        expect(isGeluDetailSelection({
            label: 'Residual Stream Vector',
            stage: 'residual.post_mlp'
        })).toBe(false);
    });
});
