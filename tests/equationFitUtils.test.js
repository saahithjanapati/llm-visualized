// @vitest-environment jsdom

import { describe, expect, it } from 'vitest';
import { readEquationContentSize } from '../src/ui/equationFitUtils.js';

function setElementBox(el, {
    width = 0,
    height = 0,
    scrollWidth = width,
    scrollHeight = height
} = {}) {
    Object.defineProperty(el, 'scrollWidth', {
        configurable: true,
        get: () => scrollWidth
    });
    Object.defineProperty(el, 'scrollHeight', {
        configurable: true,
        get: () => scrollHeight
    });
    el.getBoundingClientRect = () => ({
        width,
        height,
        left: 0,
        right: width,
        top: 0,
        bottom: height,
        x: 0,
        y: 0,
        toJSON() {
            return this;
        }
    });
    return el;
}

function appendKatexLine(container, width, height) {
    const lineEl = document.createElement('div');
    lineEl.className = 'detail-preview-equation';

    const displayEl = document.createElement('div');
    displayEl.className = 'katex-display';

    const rootEl = document.createElement('span');
    rootEl.className = 'katex';

    setElementBox(rootEl, { width, height });
    setElementBox(displayEl, { width, height });
    setElementBox(lineEl, { width, height });

    displayEl.appendChild(rootEl);
    lineEl.appendChild(displayEl);
    container.appendChild(lineEl);
    return lineEl;
}

describe('equationFitUtils', () => {
    it('measures stacked preview equations using the widest line', () => {
        const container = document.createElement('div');
        appendKatexLine(container, 132, 24);
        appendKatexLine(container, 184, 28);

        expect(readEquationContentSize(container)).toEqual({
            width: 184,
            height: 52
        });
    });

    it('falls back to the container box when no KaTeX markup is present', () => {
        const container = document.createElement('div');
        setElementBox(container, {
            width: 96,
            height: 18,
            scrollWidth: 110,
            scrollHeight: 22
        });

        expect(readEquationContentSize(container)).toEqual({
            width: 110,
            height: 22
        });
    });
});
