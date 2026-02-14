import * as THREE from 'three';
import { appState } from '../state/appState.js';
import { getPreference } from '../utils/preferences.js';
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MLP_UP_MATRIX_COLOR,
    MLP_DOWN_MATRIX_COLOR,
    MHSA_MATRIX_INITIAL_RESTING_COLOR,
    TOP_EMBED_BASE_EMISSIVE,
    TOP_EMBED_MAX_EMISSIVE
} from '../animations/LayerAnimationConstants.js';
import { USE_PHYSICAL_MATERIALS } from '../utils/constants.js';
import { applyPhysicalMaterialsToScene } from '../utils/materialUtils.js';

// Initializes status overlay and equations panel updates.
export function initStatusOverlay(pipeline, NUM_LAYERS) {
    const statusDiv = document.getElementById('statusOverlay');
    const equationsPanel = document.getElementById('equationsPanel');
    const equationsTitle = document.getElementById('equationsTitle');
    const equationsBody = document.getElementById('equationsBody');
    const shouldShowEquations = () => appState.showEquations && !appState.equationsSuppressed;
    const scheduleFrame = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
        ? window.requestAnimationFrame.bind(window)
        : (cb) => setTimeout(cb, 16);

    appState.showEquations = getPreference('showEquations', true);
    appState.showHdrBackground = getPreference('showHdrBackground', false);
    if (equationsPanel) equationsPanel.style.display = shouldShowEquations() ? 'block' : 'none';
    appState.applyEnvironmentBackground(pipeline);

    const colorHex = (hex) => `#${Number(hex).toString(16).padStart(6, '0')}`;
    const colorize = (hex, body) => `\\textcolor{${hex}}{${body}}`;
    const qColor = colorHex(MHA_FINAL_Q_COLOR);
    const kColor = colorHex(MHA_FINAL_K_COLOR);
    const vColor = colorHex(MHA_FINAL_V_COLOR);
    const woColor = colorHex(MHA_OUTPUT_PROJECTION_MATRIX_COLOR);
    const mlpUpColor = colorHex(MLP_UP_MATRIX_COLOR);
    const mlpDownColor = colorHex(MLP_DOWN_MATRIX_COLOR);
    const Q = colorize(qColor, 'Q');
    const K = colorize(kColor, 'K');
    const V = colorize(vColor, 'V');
    const WQ = colorize(qColor, 'W^Q');
    const WK = colorize(kColor, 'W^K');
    const WV = colorize(vColor, 'W^V');
    const BQ = colorize(qColor, 'b^Q');
    const BK = colorize(kColor, 'b^K');
    const BV = colorize(vColor, 'b^V');
    const WO = colorize(woColor, 'W^O');
    const BO = colorize(woColor, 'b^O');
    const WUpRaw = 'W_{\\text{up}}';
    const WDownRaw = 'W_{\\text{down}}';
    const BUpRaw = 'b_{\\text{up}}';
    const BDownRaw = 'b_{\\text{down}}';
    const WUp = colorize(mlpUpColor, WUpRaw);
    const WDown = colorize(mlpDownColor, WDownRaw);
    const BUp = colorize(mlpUpColor, BUpRaw);
    const BDown = colorize(mlpDownColor, BDownRaw);
    const U = 'u';
    const U_LN = `${U}_{\\text{ln}}`;
    const X_OUT = 'x_{\\text{out}}';
    const X_FINAL = 'x_{\\text{final}}';
    const LOGITS = '\\ell';
    const X_TOK = 'x_t^{\\text{tok}}';
    const X_POS = 'x_t^{\\text{pos}}';
    const TOK_ID = '\\mathrm{token}_t';
    const topEmbedBaseColor = new THREE.Color(MHSA_MATRIX_INITIAL_RESTING_COLOR);
    const topEmbedTargetColor = new THREE.Color(MHA_FINAL_Q_COLOR);
    const topEmbedWorkingColor = new THREE.Color();
    const topEmbedMaxEmissive = TOP_EMBED_MAX_EMISSIVE;

    const LN_EQ_BASE_COLOR = '#6a6a6a';
    const LN_EQ_ACTIVE_COLOR = '#ffffff';
    const EQ_PROGRESS_STEPS = 14;
    const eqBaseColor = new THREE.Color(LN_EQ_BASE_COLOR);
    const eqActiveColor = new THREE.Color(LN_EQ_ACTIVE_COLOR);
    const eqWorkingColor = new THREE.Color();
    const clamp01 = (value) => Math.max(0, Math.min(1, value));
    const normalizeHighlight = (value) => {
        if (Number.isFinite(value)) return clamp01(value);
        return value ? 1 : 0;
    };
    const quantizeHighlight = (value) => {
        const t = normalizeHighlight(value);
        const snapped = Math.round(t * EQ_PROGRESS_STEPS) / EQ_PROGRESS_STEPS;
        return clamp01(snapped);
    };
    const eqColorFor = (value) => {
        const t = clamp01(value);
        eqWorkingColor.copy(eqBaseColor).lerp(eqActiveColor, t);
        return `#${eqWorkingColor.getHexString()}`;
    };
    const buildLayerNormEquation = (inputSymbol, outputSymbol, highlights) => {
        const normT = normalizeHighlight(highlights.norm);
        const scaleT = normalizeHighlight(highlights.scale);
        const shiftT = normalizeHighlight(highlights.shift);
        const lhsExpr = colorize(eqColorFor(1), outputSymbol);
        const eqExpr = colorize(eqColorFor(1), '=');
        const normExpr = colorize(eqColorFor(normT), `\\frac{${inputSymbol} - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}`);
        const scaleExpr = colorize(eqColorFor(scaleT), `\\odot \\gamma`);
        const shiftExpr = colorize(eqColorFor(shiftT), `+ \\beta`);
        const body = `${lhsExpr} ${eqExpr} ${normExpr} ${scaleExpr} ${shiftExpr}`;
        return colorize(LN_EQ_BASE_COLOR, body);
    };

    const buildMlpEquation = (highlights) => {
        const upT = normalizeHighlight(highlights.up);
        const geluT = normalizeHighlight(highlights.gelu);
        const downT = normalizeHighlight(highlights.down);
        const lhsPre = colorize(eqColorFor(1), 'a');
        const lhsExpr = colorize(eqColorFor(1), 'z');
        const lhsMlp = colorize(eqColorFor(1), `\\mathrm{MLP}(${U_LN})`);
        const eqExpr = colorize(eqColorFor(1), '=');
        const upTerm = `${U_LN} ${WUpRaw} + ${BUpRaw}`;
        const upExpr = colorize(eqColorFor(upT), upTerm);
        const geluExpr = geluT > 0
            ? colorize(eqColorFor(geluT), `\\mathrm{GELU}(${upTerm})`)
            : `\\mathrm{GELU}(${upExpr})`;
        const downTerm = `z ${WDownRaw} + ${BDownRaw}`;
        const downExpr = colorize(eqColorFor(downT), downTerm);
        const body = String.raw`\begin{aligned} ${lhsPre} &${eqExpr} ${upExpr} \\ ${lhsExpr} &${eqExpr} ${geluExpr} \\ ${lhsMlp} &${eqExpr} ${downExpr} \end{aligned}`;
        return colorize(LN_EQ_BASE_COLOR, body);
    };

    const QKV_PROJECTION_EQ = String.raw`\begin{aligned} ${Q} &= x_{\text{ln}} ${WQ} + ${BQ} \\ ${K} &= x_{\text{ln}} ${WK} + ${BK} \\ ${V} &= x_{\text{ln}} ${WV} + ${BV} \end{aligned}`;
    const EQ = {
        qkv_per_head: QKV_PROJECTION_EQ,
        qkv_packed: QKV_PROJECTION_EQ,
        attn: `H_i = \\mathrm{softmax}\\left(\\frac{${Q}_i ${K}_i^\\top}{\\sqrt{d_h}} + M\\right)${V}_i,\\; i=1\\dots 12`,
        concat_proj: String.raw`\begin{aligned} H &= \mathrm{Concat}(H_i)_{i=1}^{12} \\ O &= H ${WO} + ${BO} \end{aligned}`,
        resid1: `${U} = x + O`,
        mlp: String.raw`\begin{aligned} a &= ${U_LN} ${WUp} + ${BUp} \\ z &= \mathrm{GELU}(a) \\ \mathrm{MLP}(${U_LN}) &= z ${WDown} + ${BDown} \end{aligned}`,
        resid2: String.raw`x_{\text{out}} = ${U} + \mathrm{MLP}(${U_LN})`,
        embed_token: `${X_TOK} = E[${TOK_ID}]`,
        embed_pos: `${X_POS} = P[t]`,
        embed_sum: `x_t = ${X_TOK} + ${X_POS}`,
        logits: String.raw`\begin{aligned} ${LOGITS} &= ${X_FINAL} W_U \\ p &= \mathrm{softmax}(${LOGITS}) \end{aligned}`
    };

    const EQUATION_FONT_MIN_PX = 8;
    const EQUATION_FONT_MAX_PX = 19;
    const EQUATION_FONT_MAX_SCALE = 1.45;
    const EQUATION_VERTICAL_GUARD_PX = 4;
    const EQUATION_FIT_BUFFER_PX = 1.25;
    const EQUATION_SIZE_CAPS = {
        default: { maxPx: EQUATION_FONT_MAX_PX, maxScale: EQUATION_FONT_MAX_SCALE },
        ln1: { maxPx: 17.2, maxScale: 1.32 },
        ln2: { maxPx: 17.2, maxScale: 1.32 },
        ln_top: { maxPx: 17.2, maxScale: 1.32 },
        attn: { maxPx: 16.2, maxScale: 1.2 },
        concat_proj: { maxPx: 17.0, maxScale: 1.28 },
        resid1: { maxPx: 16.8, maxScale: 1.24 }
    };
    const resolveEquationSizeCap = (eqKey) => {
        const key = typeof eqKey === 'string' ? eqKey : '';
        return EQUATION_SIZE_CAPS[key] || EQUATION_SIZE_CAPS.default;
    };
    const eqFitState = {
        baseFontPx: null,
        lastFontPx: null,
        scheduled: false,
        pending: false
    };
    const getPx = (value) => {
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : 0;
    };
    const readBaseFontPx = () => {
        if (!equationsBody || typeof window === 'undefined') return 14;
        const previous = equationsBody.style.fontSize;
        equationsBody.style.fontSize = '';
        const base = getPx(window.getComputedStyle(equationsBody).fontSize) || 14;
        equationsBody.style.fontSize = previous;
        return base;
    };
    const readEquationContentSize = () => {
        if (!equationsBody) return { width: 0, height: 0 };
        const katexDisplay = equationsBody.querySelector('.katex-display');
        const katexRoot = equationsBody.querySelector('.katex-display > .katex');
        const measureKatexBaseBounds = () => {
            if (!katexRoot) return null;
            const bases = katexRoot.querySelectorAll('.katex-html .base');
            if (!bases || !bases.length) return null;
            let left = Infinity;
            let right = -Infinity;
            let top = Infinity;
            let bottom = -Infinity;
            bases.forEach((base) => {
                const rect = base.getBoundingClientRect();
                if (!(rect.width > 0 && rect.height > 0)) return;
                left = Math.min(left, rect.left);
                right = Math.max(right, rect.right);
                top = Math.min(top, rect.top);
                bottom = Math.max(bottom, rect.bottom);
            });
            if (!Number.isFinite(left) || !Number.isFinite(right) || !Number.isFinite(top) || !Number.isFinite(bottom)) {
                return null;
            }
            return { width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
        };
        const baseBounds = measureKatexBaseBounds();
        if (baseBounds && katexRoot) {
            const rootRect = katexRoot.getBoundingClientRect();
            return {
                width: Math.max(0, baseBounds.width + 1),
                height: Math.max(0, baseBounds.height, katexRoot.scrollHeight, rootRect.height || 0)
            };
        }
        if (katexRoot) {
            const rect = katexRoot.getBoundingClientRect();
            return {
                width: Math.max(0, katexRoot.scrollWidth, rect.width || 0),
                height: Math.max(0, katexRoot.scrollHeight, rect.height || 0)
            };
        }
        if (katexDisplay) {
            const rect = katexDisplay.getBoundingClientRect();
            return {
                width: Math.max(0, katexDisplay.scrollWidth, rect.width || 0),
                height: Math.max(0, katexDisplay.scrollHeight, rect.height || 0)
            };
        }
        return {
            width: equationsBody.scrollWidth,
            height: equationsBody.scrollHeight
        };
    };
    const applyEquationFit = () => {
        if (!equationsPanel || !equationsBody) return;
        if (!shouldShowEquations()) return;
        const panelRect = equationsPanel.getBoundingClientRect();
        if (!(panelRect.width > 0 && panelRect.height > 0)) return;

        const titleStyle = equationsTitle ? window.getComputedStyle(equationsTitle) : null;
        const titleVisible = !!titleStyle && titleStyle.display !== 'none';
        const titleRect = (equationsTitle && titleVisible) ? equationsTitle.getBoundingClientRect() : { height: 0 };
        const titleMarginBottom = titleVisible ? getPx(titleStyle.marginBottom) : 0;
        const bodyStyle = window.getComputedStyle(equationsBody);
        const paddingX = getPx(bodyStyle.paddingLeft) + getPx(bodyStyle.paddingRight);
        const paddingY = getPx(bodyStyle.paddingTop) + getPx(bodyStyle.paddingBottom);

        const availableWidth = Math.max(0, panelRect.width - paddingX - 1);
        const availableHeight = Math.max(
            0,
            panelRect.height - titleRect.height - titleMarginBottom - paddingY - EQUATION_VERTICAL_GUARD_PX
        );
        if (!(availableWidth > 0 && availableHeight > 0)) return;
        const fitWidth = Math.max(0, availableWidth - EQUATION_FIT_BUFFER_PX);
        const fitHeight = Math.max(0, availableHeight - EQUATION_FIT_BUFFER_PX);
        if (!(fitWidth > 0 && fitHeight > 0)) return;

        const baseFontPx = readBaseFontPx();
        if (eqFitState.baseFontPx === null || Math.abs(baseFontPx - eqFitState.baseFontPx) > 0.5) {
            eqFitState.baseFontPx = baseFontPx;
            eqFitState.lastFontPx = null;
        }
        const clampFontPx = (value, ceiling) => {
            const maxPx = Math.max(EQUATION_FONT_MIN_PX, ceiling);
            return Math.min(maxPx, Math.max(EQUATION_FONT_MIN_PX, value));
        };
        const applyFontPx = (fontPx) => {
            equationsBody.style.fontSize = `${fontPx.toFixed(2)}px`;
        };
        const fitsAt = (fontPx) => {
            applyFontPx(fontPx);
            const fittedSize = readEquationContentSize();
            return {
                fits: fittedSize.width <= fitWidth + 0.5
                    && fittedSize.height <= fitHeight + 0.5,
                size: fittedSize
            };
        };

        const activeEqKey = typeof appState.lastEqKey === 'string' ? appState.lastEqKey : '';
        const cap = resolveEquationSizeCap(activeEqKey);
        const capPx = Number.isFinite(cap.maxPx) ? cap.maxPx : EQUATION_FONT_MAX_PX;
        const capScale = Number.isFinite(cap.maxScale) ? cap.maxScale : EQUATION_FONT_MAX_SCALE;
        const maxFontPx = Math.max(
            EQUATION_FONT_MIN_PX,
            Math.min(
                availableHeight,
                capPx,
                eqFitState.baseFontPx * capScale
            )
        );
        let low = clampFontPx(EQUATION_FONT_MIN_PX, maxFontPx);
        const minFit = fitsAt(low);
        if (!minFit.fits) {
            if (eqFitState.lastFontPx === null || Math.abs(eqFitState.lastFontPx - low) >= 0.1) {
                applyFontPx(low);
                eqFitState.lastFontPx = low;
            }
            return;
        }

        let high = clampFontPx(maxFontPx, maxFontPx);
        const highFit = fitsAt(high);
        if (!highFit.fits) {
            for (let pass = 0; pass < 9; pass += 1) {
                const mid = (low + high) * 0.5;
                const probe = fitsAt(mid);
                if (probe.fits) {
                    low = mid;
                } else {
                    high = mid;
                }
            }
        } else {
            low = high;
        }

        const targetFontPx = clampFontPx(low, maxFontPx);
        if (eqFitState.lastFontPx !== null && Math.abs(targetFontPx - eqFitState.lastFontPx) < 0.1) {
            applyFontPx(eqFitState.lastFontPx);
            return;
        }
        applyFontPx(targetFontPx);
        eqFitState.lastFontPx = targetFontPx;
    };
    const scheduleEquationFit = () => {
        if (eqFitState.scheduled) {
            eqFitState.pending = true;
            return;
        }
        eqFitState.scheduled = true;
        eqFitState.pending = false;
        scheduleFrame(() => {
            eqFitState.scheduled = false;
            if (eqFitState.pending) {
                eqFitState.pending = false;
                scheduleEquationFit();
                return;
            }
            applyEquationFit();
        });
    };

    if (equationsPanel && typeof ResizeObserver !== 'undefined') {
        const eqObserver = new ResizeObserver(() => scheduleEquationFit());
        eqObserver.observe(equationsPanel);
    }
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener('resize', scheduleEquationFit);
        if (typeof document !== 'undefined' && document.fonts?.ready) {
            document.fonts.ready.then(() => scheduleEquationFit());
        }
    }

    function renderEq(tex, title) {
        if (!equationsPanel || !equationsBody) return;
        if (!shouldShowEquations()) return;
        equationsTitle.textContent = title || 'Equations';
        if (window.katex?.render) {
            try {
                equationsBody.innerHTML = '';
                window.katex.render(tex, equationsBody, { throwOnError: false, displayMode: true });
                scheduleEquationFit();
            } catch (err) {
                console.error('KaTeX render failed:', err);
                equationsBody.textContent = tex;
                scheduleEquationFit();
            }
        } else {
            equationsBody.textContent = tex;
            scheduleEquationFit();
        }
    }

    appState.lastEqKey = '';
    appState.lastEqSignature = '';

    const SHIFT_DELAY_FRACTION = 0.2;
    const applyShiftDelay = (progress) => {
        const t = clamp01(progress);
        if (t <= SHIFT_DELAY_FRACTION) return 0;
        return clamp01((t - SHIFT_DELAY_FRACTION) / (1 - SHIFT_DELAY_FRACTION));
    };

    const getShiftProgress = (lanes, progressKey, startedKey, completeKey) => {
        let maxProgress = 0;
        for (const lane of lanes) {
            if (!lane) continue;
            const raw = lane[progressKey];
            if (Number.isFinite(raw)) {
                maxProgress = Math.max(maxProgress, raw);
                continue;
            }
            if (lane[completeKey]) {
                maxProgress = 1;
                continue;
            }
            if (lane[startedKey]) {
                maxProgress = Math.max(maxProgress, 0);
            }
        }
        return applyShiftDelay(maxProgress);
    };

    const getLayerNormHighlights = (lanes, kind) => {
        if (kind === 'ln1') {
            const norm = lanes.some(l => l?.normAnim?.isAnimating || l?.normApplied) ? 1 : 0;
            const scale = lanes.some(l => l?.multStarted) ? 1 : 0;
            const shift = getShiftProgress(lanes, 'ln1ShiftProgress', 'ln1AddStarted', 'ln1AddComplete');
            return { norm, scale, shift };
        }
        const norm = lanes.some(l => l?.normAnimationLN2?.isAnimating || l?.normAppliedLN2) ? 1 : 0;
        const scale = lanes.some(l => l?.multDoneLN2) ? 1 : 0;
        const shift = getShiftProgress(lanes, 'ln2ShiftProgress', 'ln2AddStarted', 'ln2AddComplete');
        return { norm, scale, shift };
    };

    const getMlpHighlights = (lanes) => {
        const up = lanes.some(l => l?.mlpUpStarted);
        const gelu = lanes.some(l => l?.mlpGeluActive || l?.mlpGeluComplete);
        const down = lanes.some(l => l?.mlpDownStarted || l?.mlpDownComplete);
        return { up, gelu, down };
    };

    const getTopLayerNormHighlights = (lanes) => {
        const norm = lanes.some(l => l?.__topLnEntered || l?.__topLnMultStarted || l?.__topLnShiftStarted || l?.__topLnShiftComplete);
        const scale = lanes.some(l => l?.__topLnMultStarted || l?.__topLnShiftStarted || l?.__topLnShiftComplete);
        const shift = lanes.some(l => l?.__topLnShiftStarted || l?.__topLnShiftComplete);
        return { norm, scale, shift };
    };

    const resolveInputEmbeddingStage = (layer, lanes) => {
        if (!layer || layer.index !== 0 || !Array.isArray(lanes) || !lanes.length) return null;
        let hasEmbeddingLanes = false;
        let anyTokenPassActive = false;
        let anyPosPassActive = false;
        let anySumActive = false;
        let anyPendingPosPass = false;

        for (const lane of lanes) {
            if (!lane || !lane.posVec) continue;
            hasEmbeddingLanes = true;
            if (lane.posAddComplete) continue;

            const branchY = lane.branchStartY;
            const vocabY = lane?.originalVec?.group?.position?.y;
            if (Number.isFinite(branchY) && Number.isFinite(vocabY) && vocabY < branchY - 0.25) {
                anyTokenPassActive = true;
            }

            if (!lane.__posPassStarted) {
                anyPendingPosPass = true;
                continue;
            }

            if (lane.posAddStarted) {
                anySumActive = true;
            } else {
                anyPosPassActive = true;
            }
        }

        if (!hasEmbeddingLanes) return null;
        if (anySumActive) return { eqKey: 'embed_sum', eqTitle: 'Input Embedding Sum' };
        if (anyPosPassActive) return { eqKey: 'embed_pos', eqTitle: 'Positional Embedding' };
        if (anyTokenPassActive || (anyPendingPosPass && layer._ln1Start !== true)) {
            return { eqKey: 'embed_token', eqTitle: 'Token Embedding Lookup' };
        }
        return null;
    };

    const getTopEmbeddingStats = (lastLayer) => {
        if (!lastLayer) return null;
        const entryY = Number.isFinite(lastLayer.__topEmbedEntryYLocal)
            ? lastLayer.__topEmbedEntryYLocal
            : lastLayer.__topEmbedStopYLocal;
        const exitY = Number.isFinite(lastLayer.__topEmbedExitYLocal)
            ? lastLayer.__topEmbedExitYLocal
            : entryY;
        const lanes = Array.isArray(lastLayer.lanes) ? lastLayer.lanes : [];
        const laneCount = lanes.length;
        let anyY = false;
        let highestY = -Infinity;
        let allAtEntry = laneCount > 0 && Number.isFinite(entryY);
        let allAtExit = laneCount > 0 && Number.isFinite(exitY);

        for (const lane of lanes) {
            const y = lane?.originalVec?.group?.position?.y;
            if (!Number.isFinite(y)) {
                allAtEntry = false;
                allAtExit = false;
                continue;
            }
            anyY = true;
            highestY = Math.max(highestY, y);
            if (Number.isFinite(entryY) && y < entryY - 0.5) {
                allAtEntry = false;
            }
            if (Number.isFinite(exitY) && y < exitY - 0.5) {
                allAtExit = false;
            }
        }

        return { entryY, exitY, anyY, highestY, allAtEntry, allAtExit };
    };

    const resolveFinalStage = () => {
        const lastLayer = pipeline?._layers?.[NUM_LAYERS - 1];
        if (!lastLayer) return null;
        const lanes = Array.isArray(lastLayer.lanes) ? lastLayer.lanes : [];
        const topLnActive = lanes.some(l => l?.__topLnEntered || l?.__topLnMultStarted || l?.__topLnShiftStarted || l?.__topLnShiftComplete);
        const stats = getTopEmbeddingStats(lastLayer);
        const forwardComplete = typeof pipeline?.isForwardPassComplete === 'function'
            ? pipeline.isForwardPassComplete()
            : false;
        const projectionActive = Boolean(
            forwardComplete
            || stats?.allAtExit
            || stats?.allAtEntry
            || (stats?.anyY && Number.isFinite(stats?.entryY) && stats.highestY >= stats.entryY - 0.5)
        );

        if (projectionActive) {
            return { layer: lastLayer, eqKey: 'logits', eqTitle: 'Output Logits', status: 'Output Logits', active: true };
        }
        if (topLnActive) {
            return { layer: lastLayer, eqKey: 'ln_top', eqTitle: 'Final LayerNorm', status: 'Final LayerNorm', active: true };
        }
        return { layer: lastLayer, eqKey: 'ln_top', eqTitle: 'Final LayerNorm', status: 'Final LayerNorm', active: false };
    };

    function updateEquations(layer, override = null) {
        if (!shouldShowEquations()) return;
        const targetLayer = override?.layer ?? layer;
        if (!targetLayer && !override?.eqKey) return;
        const lanes = Array.isArray(targetLayer?.lanes) ? targetLayer.lanes : [];

        if (override?.eqKey) {
            let eqBody = EQ[override.eqKey] || '';
            let signature = override.eqKey;
            if (override.eqKey === 'ln_top') {
                const highlights = getTopLayerNormHighlights(lanes);
                const normT = quantizeHighlight(highlights.norm);
                const scaleT = quantizeHighlight(highlights.scale);
                const shiftT = quantizeHighlight(highlights.shift);
                eqBody = buildLayerNormEquation(X_OUT, X_FINAL, { norm: normT, scale: scaleT, shift: shiftT });
                signature = `${override.eqKey}|n${normT.toFixed(3)}|s${scaleT.toFixed(3)}|h${shiftT.toFixed(3)}`;
            }
            if (signature === appState.lastEqSignature) return;
            appState.lastEqKey = override.eqKey;
            appState.lastEqSignature = signature;
            renderEq(eqBody, override.eqTitle);
            return;
        }

        if (!targetLayer) return;
        if (!lanes.length) return;

        const inputEmbeddingStage = resolveInputEmbeddingStage(targetLayer, lanes);
        if (inputEmbeddingStage?.eqKey) {
            const key = inputEmbeddingStage.eqKey;
            const signature = key;
            if (signature === appState.lastEqSignature) return;
            appState.lastEqKey = key;
            appState.lastEqSignature = signature;
            renderEq(EQ[key] || '', inputEmbeddingStage.eqTitle);
            return;
        }

        const resid2Active = lanes.some(l => l && l.stopRise && l.ln2Phase === 'done');
        const resid1Active = !resid2Active && lanes.some(l => {
            if (!l) return false;
            if (l.stopRise && ['travelMHSA','finishedHeads','postMHSAAddition'].includes(l.horizPhase)) {
                return true;
            }
            return l.horizPhase === 'postMHSAAddition' && l.ln2Phase === 'preRise';
        });
        const mlpActive = !resid2Active && lanes.some(l => l.mlpUpStarted || l.ln2Phase === 'mlpReady' || l.ln2Phase === 'done');
        const ln2Active = !resid2Active && !mlpActive && lanes.some(l => l.ln2Phase && l.ln2Phase !== 'notStarted');
        const ln1Active = !resid2Active && !mlpActive && !ln2Active && lanes.some(l => ['waiting','right','insideLN'].includes(l.horizPhase));
        const mhsaActive = !resid2Active && !mlpActive && !ln2Active && !ln1Active && (
            (targetLayer && targetLayer._mhsaStart === true) ||
            lanes.some(l => ['riseAboveLN','readyMHSA','travelMHSA','postMHSAAddition','waitingForLN2'].includes(l.horizPhase))
        );

        let key = '';
        let title = '';
        if (resid2Active) {
            key = 'resid2';
            title = 'Residual Add 2';
        } else if (resid1Active) {
            key = 'resid1';
            title = 'Residual Add 1';
        } else if (mlpActive) {
            const adding = lanes.some(l => l.ln2Phase === 'done' && !l.additionComplete);
            key = adding ? 'resid2' : 'mlp';
            title = adding ? 'Residual Add 2' : 'MLP (FFN)';
        } else if (ln2Active) {
            key = 'ln2';
            title = 'LayerNorm 2';
        } else if (mhsaActive) {
            const m = targetLayer.mhsaAnimation;
            const phase = m && m.mhaPassThroughPhase;
            const outPhase = m && m.outputProjMatrixAnimationPhase;
            const postAdd = lanes.some(l => ['postMHSAAddition','waitingForLN2'].includes(l.horizPhase));
            const selfAttentionActive = Boolean(
                m?.enableSelfAttentionAnimation
                && m?.selfAttentionAnimator
                && (
                    m.selfAttentionAnimator.phase === 'running'
                    || (typeof m.selfAttentionAnimator.isConveyorActive === 'function'
                        && m.selfAttentionAnimator.isConveyorActive())
                )
            );
            const concatActive = m?.rowMergePhase === 'merging'
                || outPhase === 'vectors_entering'
                || outPhase === 'vectors_inside'
                || outPhase === 'completed';

            if (postAdd) {
                key = 'resid1';
                title = 'Residual Add 1';
            } else if (selfAttentionActive) {
                key = 'attn';
                title = 'Scaled Dot-Product Attention';
            } else if (concatActive) {
                key = 'concat_proj';
                title = 'Concat Heads + W^O';
            } else if (phase === 'parallel_pass_through_active' || phase === 'mha_pass_through_complete') {
                key = 'attn';
                title = 'Scaled Dot-Product Attention';
            } else {
                key = 'qkv_per_head';
                title = 'Q/K/V Projections';
            }
        } else if (ln1Active || targetLayer.isActive) {
            key = 'ln1';
            title = 'LayerNorm 1';
        }
        if (!key) return;
        let eqBody = EQ[key];
        let signature = key;
        if (key === 'ln1') {
            const highlights = getLayerNormHighlights(lanes, 'ln1');
            const normT = quantizeHighlight(highlights.norm);
            const scaleT = quantizeHighlight(highlights.scale);
            const shiftT = quantizeHighlight(highlights.shift);
            eqBody = buildLayerNormEquation('x', 'x_{\\text{ln}}', { norm: normT, scale: scaleT, shift: shiftT });
            signature = `${key}|n${normT.toFixed(3)}|s${scaleT.toFixed(3)}|h${shiftT.toFixed(3)}`;
        } else if (key === 'ln2') {
            const highlights = getLayerNormHighlights(lanes, 'ln2');
            const normT = quantizeHighlight(highlights.norm);
            const scaleT = quantizeHighlight(highlights.scale);
            const shiftT = quantizeHighlight(highlights.shift);
            eqBody = buildLayerNormEquation(U, U_LN, { norm: normT, scale: scaleT, shift: shiftT });
            signature = `${key}|n${normT.toFixed(3)}|s${scaleT.toFixed(3)}|h${shiftT.toFixed(3)}`;
        } else if (key === 'mlp') {
            const highlights = getMlpHighlights(lanes);
            eqBody = buildMlpEquation(highlights);
            signature = `${key}|u${highlights.up ? 1 : 0}|g${highlights.gelu ? 1 : 0}|d${highlights.down ? 1 : 0}`;
        }
        if (signature === appState.lastEqSignature) return;
        appState.lastEqKey = key;
        appState.lastEqSignature = signature;
        const t = (key === 'ln1') ? 'LayerNorm 1 (no in-place assign)' : title;
        renderEq(eqBody, t);
    }

    function checkTopEmbeddingActivation() {
        const lastLayer = pipeline._layers[NUM_LAYERS - 1];
        if (!lastLayer || !appState.vocabTopRef) return;
        const entryY = Number.isFinite(lastLayer.__topEmbedEntryYLocal)
            ? lastLayer.__topEmbedEntryYLocal
            : lastLayer.__topEmbedStopYLocal;
        if (typeof entryY !== 'number') return;
        const exitY = Number.isFinite(lastLayer.__topEmbedExitYLocal)
            ? lastLayer.__topEmbedExitYLocal
            : entryY;
        const lanes = Array.isArray(lastLayer.lanes) ? lastLayer.lanes : [];
        let highestY = -Infinity;
        for (const lane of lanes) {
            const v = lane && lane.originalVec;
            if (!v || !v.group) continue;
            const y = v.group.position.y;
            if (Number.isFinite(y) && y > highestY) highestY = y;
        }
        if (!Number.isFinite(highestY)) return;
        let t = 0;
        if (highestY >= entryY) {
            const denom = Math.max(1e-6, exitY - entryY);
            t = denom > 0 ? Math.min(1, (highestY - entryY) / denom) : 1;
        }
        const eased = t * t * (3 - 2 * t);
        if (!appState.topEmbedActivated || eased >= 1) {
            topEmbedWorkingColor.copy(topEmbedBaseColor).lerp(topEmbedTargetColor, eased);
            appState.vocabTopRef.setColor(topEmbedWorkingColor);
            appState.vocabTopRef.setMaterialProperties({
                emissiveIntensity: TOP_EMBED_BASE_EMISSIVE + topEmbedMaxEmissive * eased
            });
        }
        if (eased >= 1) {
            appState.topEmbedActivated = true;
        }
    }

    let lastStatusText = '';

    function updateStatus() {
        if (!statusDiv) return;
        const idx = Number.isFinite(pipeline._currentLayerIdx) ? pipeline._currentLayerIdx : 0;
        const total = NUM_LAYERS;
        const stageOverride = resolveFinalStage();
        const isFinalStage = idx >= total || stageOverride?.active;
        const layer = (isFinalStage ? stageOverride?.layer : null) ?? pipeline._layers[idx];
        let displayStage = 'Loading...';

        if (isFinalStage) {
            displayStage = stageOverride?.status || 'Output Logits';
        } else if (layer && Array.isArray(layer.lanes) && layer.lanes.length) {
            const lanes = layer.lanes;
            const mlpActive = lanes.some(l => l.mlpUpStarted || l.ln2Phase === 'mlpReady' || l.ln2Phase === 'done');
            const ln2Active = !mlpActive && lanes.some(l => l.ln2Phase && l.ln2Phase !== 'notStarted');
            const ln1Active = !mlpActive && !ln2Active && lanes.some(l => ['waiting', 'right', 'insideLN'].includes(l.horizPhase));
            const mhsaActive = !mlpActive && !ln2Active && !ln1Active && (
                (layer && layer._mhsaStart === true) ||
                lanes.some(l => ['riseAboveLN', 'readyMHSA', 'travelMHSA', 'postMHSAAddition', 'waitingForLN2'].includes(l.horizPhase))
            );
            if (mlpActive) {
                displayStage = 'Multi-Layer Perceptron Block';
            } else if (ln2Active) {
                displayStage = 'LayerNorm 2';
            } else if (ln1Active) {
                displayStage = 'LayerNorm 1';
            } else if (mhsaActive) {
                displayStage = 'Multi Head Attention';
            } else if (layer.isActive) {
                displayStage = 'LayerNorm 1';
            }
        }

        if (isFinalStage) {
            updateEquations(layer, stageOverride || { layer, eqKey: 'logits', eqTitle: 'Output Logits', status: 'Output Logits', active: true });
        } else {
            updateEquations(layer);
        }

        const showStageLine = Boolean(displayStage);
        const safeIdx = Math.max(0, Math.min(total - 1, idx));
        const headerLine = isFinalStage ? 'Output Head' : `Layer ${safeIdx + 1} / ${total}`;
        const kvCacheStatusNote = (appState.kvCacheModeEnabled && appState.kvCachePrefillActive)
            ? '\nPre-fill stage\nKV cache enabled'
            : '';
        const nextStatusText = showStageLine
            ? `${headerLine}\n${displayStage}${kvCacheStatusNote}`
            : `${headerLine}${kvCacheStatusNote}`;
        if (nextStatusText === lastStatusText) {
            checkTopEmbeddingActivation();
            return;
        }

        lastStatusText = nextStatusText;
        statusDiv.textContent = nextStatusText;
        checkTopEmbeddingActivation();
    }

    let framePending = false;
    let needsUpdate = false;

    const scheduleStatusUpdate = () => {
        if (framePending) return;
        framePending = true;
        scheduleFrame(() => {
            framePending = false;
            if (!needsUpdate) return;
            needsUpdate = false;
            updateStatus();
        });
    };

    const onProgress = () => {
        needsUpdate = true;
        scheduleStatusUpdate();
    };

    applyPhysicalMaterialsToScene(pipeline?.engine?.scene, USE_PHYSICAL_MATERIALS);
    if (pipeline && typeof pipeline.addEventListener === 'function') {
        pipeline.addEventListener('progress', onProgress);
    }
    updateStatus();

    const tickEquations = () => {
        if (shouldShowEquations() && pipeline?._layers?.length) {
            const idx = Number.isFinite(pipeline._currentLayerIdx) ? pipeline._currentLayerIdx : 0;
            const stageOverride = resolveFinalStage();
            if (idx >= NUM_LAYERS || stageOverride?.active) {
                updateEquations(stageOverride?.layer, stageOverride || { layer: pipeline._layers[NUM_LAYERS - 1], eqKey: 'logits', eqTitle: 'Output Logits', status: 'Output Logits', active: true });
            } else {
                updateEquations(pipeline._layers[idx]);
            }
        }
        scheduleFrame(tickEquations);
    };
    tickEquations();
}
