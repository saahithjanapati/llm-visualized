import { describe, expect, it } from 'vitest';

import {
    drawSimpleTex,
    resolveSimpleTexPlainText
} from './simpleTex.js';

function createMockContext() {
    return {
        operations: [],
        font: '',
        fillStyle: '',
        strokeStyle: '',
        lineWidth: 0,
        lineCap: '',
        textBaseline: '',
        save() {},
        restore() {},
        beginPath() {
            this.operations.push({ type: 'beginPath' });
        },
        moveTo(x, y) {
            this.operations.push({ type: 'moveTo', x, y });
        },
        lineTo(x, y) {
            this.operations.push({ type: 'lineTo', x, y });
        },
        stroke() {
            this.operations.push({ type: 'stroke', strokeStyle: this.strokeStyle, lineWidth: this.lineWidth });
        },
        fillText(text, x, y) {
            this.operations.push({ type: 'fillText', text, x, y, font: this.font });
        },
        measureText(text) {
            const fontSizeMatch = String(this.font || '').match(/([0-9.]+)px/);
            const fontSize = Number.parseFloat(fontSizeMatch?.[1] || '12');
            return {
                width: text.length * fontSize * 0.6
            };
        }
    };
}

describe('simpleTex', () => {
    it('resolves sqrt and text-style wrappers into readable plain text', () => {
        expect(resolveSimpleTexPlainText('\\sqrt{d_{\\mathrm{head}}}')).toBe('√dhead');
        expect(resolveSimpleTexPlainText('K^{\\mathsf{T}}')).toBe('KT');
    });

    it('draws sqrt labels with a radical stroke, radicand text, and an overline', () => {
        const ctx = createMockContext();

        drawSimpleTex(ctx, '\\sqrt{d_{\\mathrm{head}}}', {
            x: 100,
            y: 40,
            fontSize: 12,
            color: '#fff'
        });

        expect(ctx.operations.some((entry) => entry.type === 'fillText' && entry.text === 'd')).toBe(true);
        expect(ctx.operations.some((entry) => entry.type === 'fillText' && entry.text === 'head')).toBe(true);
        expect(ctx.operations.some((entry) => entry.type === 'stroke')).toBe(true);
        expect(ctx.operations.some((entry) => entry.type === 'lineTo')).toBe(true);
    });
});
