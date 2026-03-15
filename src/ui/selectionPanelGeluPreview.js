import {
    readEquationBaseFontPx,
    readEquationContentSize
} from './equationFitUtils.js';

const GELU_DOMAIN_MIN = -4;
const GELU_DOMAIN_MAX = 4;
const GELU_RANGE_MIN = -1;
const GELU_RANGE_MAX = 4;
const GELU_RELATION_EQUATION_TEX = '\\operatorname{GELU}(x) = x\\,\\Phi(x)';
const GELU_APPROXIMATION_EQUATION_TEX = '\\operatorname{GELU}(x) \\approx \\frac{x}{2}\\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}}\\left(x + 0.044715x^3\\right)\\right)\\right)';
const GELU_RELATION_FALLBACK_TEXT = 'GELU(x) = x * Phi(x)';
const GELU_APPROXIMATION_FALLBACK_TEXT = 'GELU(x) ~= 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))';
const GELU_DETAIL_EQUATION_MIN_FONT_PX = 8.25;
const GELU_DETAIL_EQUATION_FIT_BUFFER_PX = 6;
const GELU_GRAPH_MARGIN = Object.freeze({
    left: 44,
    right: 18,
    top: 18,
    bottom: 30
});

export const GELU_PANEL_ACTION_OPEN = 'open-gelu-preview';

function clamp(value, min, max) {
    if (!Number.isFinite(value)) return min;
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

function formatGeluValue(value) {
    if (!Number.isFinite(value)) return '--';
    return value.toFixed(4);
}

export function evaluateGelu(x) {
    const x3 = x * x * x;
    const inner = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x3);
    return 0.5 * x * (1 + Math.tanh(inner));
}

export function isMlpMatrixSelectionLabel(label = '') {
    const lower = String(label || '').toLowerCase();
    return lower.includes('mlp up weight matrix') || lower.includes('mlp down weight matrix');
}

export function setDescriptionGeluAction(descriptionEl, enabled = false) {
    if (!descriptionEl) return;

    descriptionEl
        .querySelectorAll(`[data-detail-action="${GELU_PANEL_ACTION_OPEN}"]`)
        .forEach((node) => {
            const row = node.closest('.detail-description-action-row');
            if (row) {
                row.remove();
                return;
            }
            node.remove();
        });

    if (!enabled) return;

    const actionRow = document.createElement('div');
    actionRow.className = 'detail-description-action-row';

    const actionBtn = document.createElement('button');
    actionBtn.type = 'button';
    actionBtn.className = 'detail-description-action-link';
    actionBtn.dataset.detailAction = GELU_PANEL_ACTION_OPEN;
    actionBtn.textContent = 'View GELU details';
    actionBtn.setAttribute('aria-label', 'Open the GELU activation detail view');

    actionRow.appendChild(actionBtn);
    descriptionEl.appendChild(actionRow);
}

function buildAxisTicks(min, max, step = 1) {
    const ticks = [];
    const safeStep = Math.max(0.25, Math.abs(step || 1));
    for (let value = min; value <= max + 1e-8; value += safeStep) {
        ticks.push(Number(value.toFixed(6)));
    }
    return ticks;
}

function toPlotX(x, width, margin) {
    const span = GELU_DOMAIN_MAX - GELU_DOMAIN_MIN;
    const usableWidth = Math.max(1, width - margin.left - margin.right);
    return margin.left + ((x - GELU_DOMAIN_MIN) / span) * usableWidth;
}

function toPlotY(y, height, margin) {
    const span = GELU_RANGE_MAX - GELU_RANGE_MIN;
    const usableHeight = Math.max(1, height - margin.top - margin.bottom);
    return margin.top + ((GELU_RANGE_MAX - y) / span) * usableHeight;
}

function toDomainX(pixelX, width, margin) {
    const usableWidth = Math.max(1, width - margin.left - margin.right);
    const t = clamp((pixelX - margin.left) / usableWidth, 0, 1);
    return GELU_DOMAIN_MIN + t * (GELU_DOMAIN_MAX - GELU_DOMAIN_MIN);
}

function renderFormulaMarkup(tex = '') {
    const safeTex = typeof tex === 'string' ? tex.trim() : '';
    if (!safeTex) return '';
    const katex = (typeof window !== 'undefined') ? window.katex : null;
    if (katex && typeof katex.renderToString === 'function') {
        try {
            return katex.renderToString(safeTex, { throwOnError: false, displayMode: true });
        } catch (_) {
            // Fall back to plain text below.
        }
    }
    return safeTex;
}

function renderFormulaBody(formulaEl, tex, fallbackText) {
    if (!formulaEl) return;
    const markup = renderFormulaMarkup(tex);
    if (markup === tex) {
        formulaEl.textContent = fallbackText || tex;
        return;
    }
    formulaEl.innerHTML = markup;
}

function fitFormulaBodyToWidth(formulaEl) {
    if (!formulaEl) return;

    const bodyRect = formulaEl.getBoundingClientRect();
    const availableWidth = Math.max(0, bodyRect.width);
    if (!(availableWidth > 0)) return;

    const fitWidth = Math.max(0, availableWidth - GELU_DETAIL_EQUATION_FIT_BUFFER_PX);
    if (!(fitWidth > 0)) return;

    const baseFontPx = readEquationBaseFontPx(formulaEl, 12);
    const maxFontPx = Math.max(GELU_DETAIL_EQUATION_MIN_FONT_PX, baseFontPx);
    const applyFontPx = (fontPx) => {
        formulaEl.style.fontSize = `${fontPx.toFixed(2)}px`;
    };
    const fitsAt = (fontPx) => {
        applyFontPx(fontPx);
        const size = readEquationContentSize(formulaEl);
        return size.width <= fitWidth + 0.5;
    };

    let low = GELU_DETAIL_EQUATION_MIN_FONT_PX;
    if (!fitsAt(low)) {
        applyFontPx(low);
        return;
    }

    let high = maxFontPx;
    if (!fitsAt(high)) {
        for (let pass = 0; pass < 9; pass += 1) {
            const mid = (low + high) * 0.5;
            if (fitsAt(mid)) {
                low = mid;
            } else {
                high = mid;
            }
        }
    } else {
        low = high;
    }

    applyFontPx(Math.max(GELU_DETAIL_EQUATION_MIN_FONT_PX, Math.min(maxFontPx, low)));
}

export function createGeluDetailView(panelEl) {
    if (!panelEl || typeof document === 'undefined') return null;

    const root = document.createElement('section');
    root.className = 'detail-gelu-view';
    root.setAttribute('aria-hidden', 'true');
    root.innerHTML = `
        <div class="detail-description detail-gelu-copy">
            <p>GELU (Gaussian Error Linear Unit) is the MLP activation used by GPT-2.</p>
            <p>Its exact form is <span class="detail-gelu-inline-equation">GELU(x) = x Phi(x)</span>, where <span class="detail-gelu-inline-equation">Phi(x)</span> is the CDF of a standard Gaussian, so it behaves like a smooth probability gate instead of a hard cutoff.</p>
        </div>
        <div class="detail-equations detail-gelu-equation-shell is-visible" aria-hidden="false">
            <div class="detail-gelu-equation-group">
                <div class="detail-gelu-equation-label">Exact relationship</div>
                <div class="detail-equations-body detail-gelu-formula" data-gelu-formula></div>
            </div>
            <p class="detail-gelu-equation-note">Here, <span class="detail-gelu-inline-equation">Phi(x)</span> is the CDF of the standard normal distribution <span class="detail-gelu-inline-equation">N(0, 1)</span>.</p>
            <div class="detail-gelu-equation-group">
                <div class="detail-gelu-equation-label">Common GPT-2 approximation</div>
                <div class="detail-equations-body detail-gelu-approx-formula" data-gelu-approx-formula></div>
            </div>
        </div>
        <div class="detail-data detail-gelu-graph-card">
            <div class="detail-gelu-card-header">
                <div class="detail-data-title">Interactive curve</div>
                <div class="detail-gelu-graph-hint">Move horizontally across the graph to inspect the tanh approximation of GELU(x).</div>
            </div>
            <canvas class="detail-gelu-graph-canvas" aria-label="Interactive GELU graph"></canvas>
        </div>
        <div class="detail-meta detail-gelu-readout" aria-label="GELU readout">
            <div class="detail-row detail-gelu-readout-card">
                <span class="detail-label detail-gelu-readout-label">Input x</span>
                <span class="detail-value detail-gelu-readout-value" data-gelu-readout="x">--</span>
            </div>
            <div class="detail-row detail-gelu-readout-card">
                <span class="detail-label detail-gelu-readout-label">Output GELU(x)</span>
                <span class="detail-value detail-gelu-readout-value" data-gelu-readout="y">--</span>
            </div>
        </div>
        <div class="detail-data detail-gelu-notes-card">
            <div class="detail-data-title">Why it matters</div>
            <ul>
                <li>The exact form is <span class="detail-gelu-inline-equation">GELU(x) = x Phi(x)</span>, so activation strength is tied to the Gaussian CDF.</li>
                <li>Near zero, outputs change smoothly instead of switching sharply.</li>
                <li>Negative values are softly damped rather than hard-clipped.</li>
                <li>Positive values pass through almost linearly at larger magnitudes.</li>
            </ul>
        </div>
    `;

    const header = panelEl.querySelector('.detail-header');
    if (header && header.nextSibling) {
        panelEl.insertBefore(root, header.nextSibling);
    } else {
        panelEl.appendChild(root);
    }

    const canvas = root.querySelector('.detail-gelu-graph-canvas');
    const formulaEl = root.querySelector('[data-gelu-formula]');
    const approxFormulaEl = root.querySelector('[data-gelu-approx-formula]');
    const readoutX = root.querySelector('[data-gelu-readout="x"]');
    const readoutY = root.querySelector('[data-gelu-readout="y"]');
    const ctx = canvas?.getContext('2d') || null;

    const state = {
        hoverX: null,
        visible: false
    };

    const xTicks = buildAxisTicks(GELU_DOMAIN_MIN, GELU_DOMAIN_MAX, 1);
    const yTicks = buildAxisTicks(GELU_RANGE_MIN, GELU_RANGE_MAX, 1);

    function renderFormula() {
        renderFormulaBody(formulaEl, GELU_RELATION_EQUATION_TEX, GELU_RELATION_FALLBACK_TEXT);
        renderFormulaBody(
            approxFormulaEl,
            GELU_APPROXIMATION_EQUATION_TEX,
            GELU_APPROXIMATION_FALLBACK_TEXT
        );
    }

    function fitFormulaBodies() {
        fitFormulaBodyToWidth(formulaEl);
        fitFormulaBodyToWidth(approxFormulaEl);
    }

    renderFormula();

    function updateReadout(x = null) {
        if (!readoutX || !readoutY) return;
        if (!Number.isFinite(x)) {
            readoutX.textContent = '--';
            readoutY.textContent = '--';
            return;
        }
        const y = evaluateGelu(x);
        readoutX.textContent = formatGeluValue(x);
        readoutY.textContent = formatGeluValue(y);
    }

    function drawGraph() {
        if (!ctx || !canvas || !state.visible) return;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, Math.floor(rect.width));
        const height = Math.max(1, Math.floor(rect.height));
        const dpr = Number.isFinite(window.devicePixelRatio) ? Math.max(1, window.devicePixelRatio) : 1;

        const targetWidth = Math.max(1, Math.round(width * dpr));
        const targetHeight = Math.max(1, Math.round(height * dpr));
        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
            canvas.width = targetWidth;
            canvas.height = targetHeight;
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        ctx.clearRect(0, 0, width, height);

        const margin = GELU_GRAPH_MARGIN;
        const plotLeft = margin.left;
        const plotTop = margin.top;
        const plotRight = width - margin.right;
        const plotBottom = height - margin.bottom;

        ctx.fillStyle = '#030303';
        ctx.fillRect(0, 0, width, height);

        ctx.fillStyle = '#070707';
        ctx.fillRect(plotLeft, plotTop, Math.max(1, plotRight - plotLeft), Math.max(1, plotBottom - plotTop));
        ctx.strokeStyle = 'rgba(255,255,255,0.12)';
        ctx.lineWidth = 1;
        ctx.strokeRect(
            plotLeft + 0.5,
            plotTop + 0.5,
            Math.max(1, plotRight - plotLeft),
            Math.max(1, plotBottom - plotTop)
        );

        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 1;
        xTicks.forEach((tick) => {
            const px = toPlotX(tick, width, margin);
            ctx.beginPath();
            ctx.moveTo(px + 0.5, plotTop);
            ctx.lineTo(px + 0.5, plotBottom);
            ctx.stroke();
        });
        yTicks.forEach((tick) => {
            const py = toPlotY(tick, height, margin);
            ctx.beginPath();
            ctx.moveTo(plotLeft, py + 0.5);
            ctx.lineTo(plotRight, py + 0.5);
            ctx.stroke();
        });

        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1.2;
        const axisY = toPlotY(0, height, margin);
        const axisX = toPlotX(0, width, margin);
        ctx.beginPath();
        ctx.moveTo(plotLeft, axisY + 0.5);
        ctx.lineTo(plotRight, axisY + 0.5);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(axisX + 0.5, plotTop);
        ctx.lineTo(axisX + 0.5, plotBottom);
        ctx.stroke();

        ctx.strokeStyle = '#dba14a';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        const sampleCount = Math.max(80, Math.round((plotRight - plotLeft) * 1.25));
        for (let idx = 0; idx <= sampleCount; idx += 1) {
            const t = idx / sampleCount;
            const x = GELU_DOMAIN_MIN + t * (GELU_DOMAIN_MAX - GELU_DOMAIN_MIN);
            const y = evaluateGelu(x);
            const px = toPlotX(x, width, margin);
            const py = toPlotY(y, height, margin);
            if (idx === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        ctx.fillStyle = 'rgba(205,213,224,0.82)';
        ctx.font = '10px "IBM Plex Mono", "SFMono-Regular", Menlo, Monaco, Consolas, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        xTicks.forEach((tick) => {
            const px = toPlotX(tick, width, margin);
            ctx.fillText(String(tick), px, plotBottom + 6);
        });

        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        yTicks.forEach((tick) => {
            const py = toPlotY(tick, height, margin);
            ctx.fillText(String(tick), plotLeft - 8, py);
        });

        if (Number.isFinite(state.hoverX)) {
            const x = clamp(
                toDomainX(state.hoverX, width, margin),
                GELU_DOMAIN_MIN,
                GELU_DOMAIN_MAX
            );
            const y = evaluateGelu(x);
            const px = toPlotX(x, width, margin);
            const py = toPlotY(y, height, margin);

            ctx.setLineDash([6, 5]);
            ctx.strokeStyle = 'rgba(255, 220, 158, 0.62)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(px, plotTop);
            ctx.lineTo(px, plotBottom);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(plotLeft, py);
            ctx.lineTo(plotRight, py);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = '#ffd79b';
            ctx.beginPath();
            ctx.arc(px, py, 4.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#120f09';
            ctx.lineWidth = 1.2;
            ctx.stroke();

            updateReadout(x);
        } else {
            updateReadout(null);
        }
    }

    function onPointerMove(event) {
        if (!state.visible || !canvas) return;
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const px = event.clientX - rect.left;
        state.hoverX = clamp(px, GELU_GRAPH_MARGIN.left, rect.width - GELU_GRAPH_MARGIN.right);
        drawGraph();
    }

    function onPointerLeave() {
        if (state.hoverX === null) return;
        state.hoverX = null;
        drawGraph();
    }

    canvas?.addEventListener('pointermove', onPointerMove);
    canvas?.addEventListener('pointerdown', onPointerMove);
    canvas?.addEventListener('pointerleave', onPointerLeave);
    canvas?.addEventListener('pointercancel', onPointerLeave);

    return {
        root,
        setVisible(visible) {
            state.visible = !!visible;
            root.classList.toggle('is-visible', state.visible);
            root.setAttribute('aria-hidden', state.visible ? 'false' : 'true');
            if (!state.visible) {
                state.hoverX = null;
                updateReadout(null);
                return;
            }
            renderFormula();
            fitFormulaBodies();
            drawGraph();
        },
        resizeAndRender() {
            fitFormulaBodies();
            drawGraph();
        },
        remove() {
            canvas?.removeEventListener('pointermove', onPointerMove);
            canvas?.removeEventListener('pointerdown', onPointerMove);
            canvas?.removeEventListener('pointerleave', onPointerLeave);
            canvas?.removeEventListener('pointercancel', onPointerLeave);
            root.remove();
        }
    };
}
