import { VIEW2D_NODE_KINDS } from './schema/sceneTypes.js';
import { resolveSimpleTexPlainText } from './simpleTex.js';

function normalizeCaptionLine(line = null) {
    if (!line || typeof line !== 'object') return null;
    const tex = typeof line.tex === 'string' ? line.tex.trim() : '';
    const text = typeof line.text === 'string' ? line.text.trim() : '';
    if (!tex.length && !text.length) return null;
    return {
        tex,
        text: text.length ? text : resolveSimpleTexPlainText(tex)
    };
}

export function resolveView2dCaptionPosition(node = null) {
    const position = String(node?.metadata?.caption?.position || '').trim().toLowerCase();
    if (position === 'top' || position === 'inside-top' || position === 'float-top') {
        return position;
    }
    return 'bottom';
}

export function resolveView2dCaptionStyleKey(node = null, fallbackStyleKey = null) {
    const styleKey = String(node?.metadata?.caption?.styleKey || '').trim();
    return styleKey.length ? styleKey : fallbackStyleKey;
}

export function resolveView2dCaptionLines(node = null) {
    if (!node || typeof node !== 'object') return [];

    const captionMeta = node.metadata?.caption || null;
    if (Array.isArray(captionMeta?.lines) && captionMeta.lines.length) {
        return captionMeta.lines
            .map((line) => normalizeCaptionLine(line))
            .filter(Boolean);
    }

    const lines = [];
    const labelLine = normalizeCaptionLine(node.label);
    if (labelLine) {
        lines.push(labelLine);
    }

    const hasExplicitDimensionCaption = (
        (typeof captionMeta?.dimensionsTex === 'string' && captionMeta.dimensionsTex.trim().length)
        || (typeof captionMeta?.dimensionsText === 'string' && captionMeta.dimensionsText.trim().length)
    );
    if (
        hasExplicitDimensionCaption
        && node.kind === VIEW2D_NODE_KINDS.MATRIX
        && Number.isFinite(node.dimensions?.rows)
        && Number.isFinite(node.dimensions?.cols)
    ) {
        const dimensionLine = normalizeCaptionLine({
            tex: captionMeta?.dimensionsTex || '',
            text: captionMeta?.dimensionsText || ''
        });
        if (dimensionLine) {
            lines.push(dimensionLine);
        }
    }

    return lines;
}

export function resolveView2dCaptionMeasurementText(line = null) {
    if (!line || typeof line !== 'object') return '';
    const texText = resolveSimpleTexPlainText(line.tex || '');
    const text = typeof line.text === 'string' ? line.text.trim() : '';
    return texText || text;
}
