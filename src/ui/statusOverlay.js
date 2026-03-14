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
    POSITION_EMBED_COLOR,
    MHSA_MATRIX_INITIAL_RESTING_COLOR,
    TOP_EMBED_BASE_EMISSIVE,
    TOP_EMBED_MAX_EMISSIVE
} from '../animations/LayerAnimationConstants.js';
import { USE_PHYSICAL_MATERIALS } from '../utils/constants.js';
import { applyPhysicalMaterialsToScene } from '../utils/materialUtils.js';
import { buildAttentionEquationSet } from './attentionEquationTextUtils.js';
import {
    readEquationBaseFontPx,
    readEquationContentSize as measureEquationContentSize
} from './equationFitUtils.js';
import { resolveMlpOverlayStage } from './statusOverlayMlpEquationUtils.js';
import {
    KV_CACHE_INFO_REQUEST_EVENT,
    buildKvCacheOverlayBadgeText,
} from './kvCacheInfoUtils.js';
import { initTouchClickFallback } from './touchClickFallback.js';
import { getTopEmbeddingActivationEasedProgress } from '../utils/topEmbeddingTimingUtils.js';

// Initializes status overlay and equations panel updates.
export function initStatusOverlay(pipeline, NUM_LAYERS) {
    const statusDiv = document.getElementById('statusOverlay');
    const equationsPanel = document.getElementById('equationsPanel');
    const equationsTitle = document.getElementById('equationsTitle');
    const equationsBody = document.getElementById('equationsBody');
    const statusTextEl = (statusDiv && typeof document !== 'undefined')
        ? document.createElement('span')
        : null;
    const statusKvCacheLink = (statusDiv && typeof document !== 'undefined')
        ? document.createElement('button')
        : null;
    const shouldShowEquations = () => appState.showEquations && !appState.equationsSuppressed;
    const scheduleFrame = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
        ? window.requestAnimationFrame.bind(window)
        : (cb) => setTimeout(cb, 16);

    appState.showEquations = getPreference('showEquations', true);
    appState.showHdrBackground = getPreference('showHdrBackground', false);
    if (equationsPanel) equationsPanel.style.display = shouldShowEquations() ? 'block' : 'none';
    appState.applyEnvironmentBackground(pipeline);

    if (statusDiv && statusTextEl && statusKvCacheLink) {
        statusDiv.textContent = '';
        statusTextEl.className = 'status-overlay__text';
        statusKvCacheLink.type = 'button';
        statusKvCacheLink.className = 'status-overlay__kv-link';
        statusKvCacheLink.hidden = true;
        statusKvCacheLink.setAttribute('aria-hidden', 'true');
        statusKvCacheLink.addEventListener('click', () => {
            if (!appState.kvCacheModeEnabled || typeof window === 'undefined') return;
            const phase = appState.kvCachePrefillActive ? 'prefill' : 'decode';
            window.dispatchEvent(new CustomEvent(KV_CACHE_INFO_REQUEST_EVENT, {
                detail: { phase }
            }));
        });
        statusDiv.append(statusTextEl, statusKvCacheLink);
        // Touchscreens occasionally miss the synthetic click on this HUD link,
        // so mirror the panel/modal fallback path and activate on pointerdown.
        initTouchClickFallback(statusDiv, {
            selector: '.status-overlay__kv-link',
            activateOnPointerDownSelector: '.status-overlay__kv-link'
        });
    }

    const colorHex = (hex) => `#${Number(hex).toString(16).padStart(6, '0')}`;
    const colorize = (hex, body) => `\\textcolor{${hex}}{${body}}`;
    const qColor = colorHex(MHA_FINAL_Q_COLOR);
    const kColor = colorHex(MHA_FINAL_K_COLOR);
    const vColor = colorHex(MHA_FINAL_V_COLOR);
    const woColor = colorHex(MHA_OUTPUT_PROJECTION_MATRIX_COLOR);
    const mlpUpColor = colorHex(MLP_UP_MATRIX_COLOR);
    const mlpDownColor = colorHex(MLP_DOWN_MATRIX_COLOR);
    const embeddingVocabColor = qColor;
    const embeddingPosColor = colorHex(POSITION_EMBED_COLOR);
    const Q = colorize(qColor, 'Q');
    const K = colorize(kColor, 'K');
    const V = colorize(vColor, 'V');
    const WQ = colorize(qColor, 'W_Q');
    const WK = colorize(kColor, 'W_K');
    const WV = colorize(vColor, 'W_V');
    const BQ = colorize(qColor, 'b_Q');
    const BK = colorize(kColor, 'b_K');
    const BV = colorize(vColor, 'b_V');
    const WO = colorize(woColor, 'W_O');
    const BO = colorize(woColor, 'b_O');
    const WUpRaw = 'W_{\\text{up}}';
    const WDownRaw = 'W_{\\text{down}}';
    const BUpRaw = 'b_{\\text{up}}';
    const BDownRaw = 'b_{\\text{down}}';
    const WUp = colorize(mlpUpColor, WUpRaw);
    const WDown = colorize(mlpDownColor, WDownRaw);
    const BUp = colorize(mlpUpColor, BUpRaw);
    const BDown = colorize(mlpDownColor, BDownRaw);
    const E = colorize(embeddingVocabColor, 'E');
    const P = colorize(embeddingPosColor, 'P');
    const WU = colorize(embeddingVocabColor, 'W_U');
    const U = 'u';
    const X_LN = 'x_{\\text{ln}}';
    const MLPResidual = `${colorize(mlpDownColor, '\\mathrm{MLP}')}(${X_LN})`;
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

    const attentionEquations = buildAttentionEquationSet({
        Q,
        K,
        V,
        WQ,
        WK,
        WV,
        BQ,
        BK,
        BV,
        WO,
        BO
    });
    const EQ = {
        qkv_per_head: attentionEquations.qkvProjection,
        qkv_packed: attentionEquations.qkvProjection,
        attn: attentionEquations.attention,
        concat_proj: attentionEquations.concatProjection,
        resid1: attentionEquations.postAttentionResidual,
        mlp_up: `a = ${X_LN} ${WUp} + ${BUp}`,
        mlp_gelu: 'z = \\mathrm{GELU}(a)',
        mlp_down: `\\mathrm{MLP}(${X_LN}) = z ${WDown} + ${BDown}`,
        resid2: String.raw`x_{\text{out}} = ${U} + ${MLPResidual}`,
        embed_token: `${X_TOK} = ${E}[${TOK_ID}]`,
        embed_pos: `${X_POS} = ${P}[t]`,
        embed_sum: `x_t = ${X_TOK} + ${X_POS}`,
        logits: String.raw`\begin{aligned} ${LOGITS} &= ${X_FINAL} ${WU} \\ p &= \mathrm{softmax}(${LOGITS}) \end{aligned}`
    };

    const EQUATION_FONT_MIN_PX = 8;
    const EQUATION_FONT_MAX_PX = 19;
    const EQUATION_FONT_MAX_SCALE = 1.45;
    const EQUATION_VERTICAL_GUARD_PX = 4;
    const EQUATION_FIT_BUFFER_PX = 1.25;
    const EQUATION_SIZE_CAPS = {
        default: { maxPx: EQUATION_FONT_MAX_PX, maxScale: EQUATION_FONT_MAX_SCALE },
        qkv_per_head: { maxPx: 15.6, maxScale: 1.14 },
        qkv_packed: { maxPx: 15.6, maxScale: 1.14 },
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
        return readEquationBaseFontPx(equationsBody, 14);
    };
    const readEquationContentSize = () => {
        return measureEquationContentSize(equationsBody);
    };
    const applyEquationFit = () => {
        if (!equationsPanel || !equationsBody) return;
        if (!shouldShowEquations()) return;
        equationsPanel.style.height = '';
        equationsBody.style.minHeight = '';
        const bodyRect = equationsBody.getBoundingClientRect();
        if (!(bodyRect.width > 0)) return;
        const panelRect = equationsPanel.getBoundingClientRect();
        const panelStyle = window.getComputedStyle(equationsPanel);
        const bodyStyle = window.getComputedStyle(equationsBody);
        const paddingX = getPx(bodyStyle.paddingLeft) + getPx(bodyStyle.paddingRight);
        const paddingY = getPx(bodyStyle.paddingTop) + getPx(bodyStyle.paddingBottom);
        const chromeHeight = Math.max(0, panelRect.height - bodyRect.height);
        const panelMinHeight = Math.max(0, getPx(panelStyle.minHeight));
        const panelMaxHeight = Math.max(
            panelMinHeight,
            getPx(panelStyle.maxHeight)
            || Math.max(panelMinHeight, (typeof window !== 'undefined' ? window.innerHeight * 0.26 : panelRect.height))
        );
        const bodyMinHeight = Math.max(0, getPx(bodyStyle.minHeight));

        const availableWidth = Math.max(0, bodyRect.width - paddingX - 1);
        const maxBodyHeight = Math.max(bodyMinHeight, panelMaxHeight - chromeHeight);
        const availableHeight = Math.max(0, maxBodyHeight - paddingY - EQUATION_VERTICAL_GUARD_PX);
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
        const applyPanelHeightForContent = (contentHeight) => {
            const desiredBodyHeight = Math.max(
                bodyMinHeight,
                contentHeight + paddingY + EQUATION_VERTICAL_GUARD_PX
            );
            const targetPanelHeight = Math.max(
                panelMinHeight,
                Math.min(panelMaxHeight, desiredBodyHeight + chromeHeight)
            );
            const targetBodyHeight = Math.max(bodyMinHeight, targetPanelHeight - chromeHeight);
            equationsPanel.style.height = `${targetPanelHeight.toFixed(2)}px`;
            equationsBody.style.minHeight = `${targetBodyHeight.toFixed(2)}px`;
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
            applyFontPx(low);
            applyPanelHeightForContent(minFit.size.height);
            eqFitState.lastFontPx = low;
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
        applyFontPx(targetFontPx);
        applyPanelHeightForContent(readEquationContentSize().height);
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
        if (window.visualViewport && typeof window.visualViewport.addEventListener === 'function') {
            window.visualViewport.addEventListener('resize', scheduleEquationFit);
        }
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

    const clearStatusText = () => {
        if (!statusTextEl) return;
        while (statusTextEl.firstChild) {
            statusTextEl.removeChild(statusTextEl.firstChild);
        }
    };

    const appendStatusLine = (text) => {
        if (!statusTextEl || typeof document === 'undefined') return;
        const safeText = String(text || '').trim();
        if (!safeText) return;
        const lineEl = document.createElement('span');
        lineEl.className = 'status-overlay__line';
        lineEl.textContent = safeText;
        statusTextEl.appendChild(lineEl);
    };

    const renderStatusText = ({ headerLine = '', stageLine = '' } = {}) => {
        if (!statusTextEl) return;
        clearStatusText();
        if (headerLine) appendStatusLine(headerLine);
        if (stageLine) appendStatusLine(stageLine);
    };

    appState.lastEqKey = '';
    appState.lastEqSignature = '';
    let eqLastLayerIndex = null;
    let eqResidualLockLayerIndex = null;

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

    const getTopLayerNormHighlights = (lanes) => {
        const norm = lanes.some(l => l?.__topLnEntered || l?.__topLnMultStarted || l?.__topLnShiftStarted || l?.__topLnShiftComplete);
        const scale = lanes.some(l => l?.__topLnMultStarted || l?.__topLnShiftStarted || l?.__topLnShiftComplete);
        const shift = lanes.some(l => l?.__topLnShiftStarted || l?.__topLnShiftComplete);
        return { norm, scale, shift };
    };

    const getNowMs = () => (
        (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now()
            : Date.now()
    );

    const hasGateTokenRiseStarted = (gate, nowMs) => {
        if (!gate || gate.enabled === false || gate.pending) return false;
        const startByToken = gate.startByToken;
        if (startByToken && typeof startByToken === 'object') {
            const starts = Object.values(startByToken).filter((value) => Number.isFinite(value));
            if (starts.length) {
                const earliestStart = Math.min(...starts);
                if (!Number.isFinite(earliestStart)) return false;
                return Number.isFinite(nowMs) ? nowMs >= earliestStart : true;
            }
        }
        const insideByToken = gate.insideByToken;
        if (insideByToken && typeof insideByToken === 'object') {
            const states = Object.values(insideByToken).filter((value) => typeof value === 'boolean');
            if (states.some((value) => value === true)) return true;
        }
        return false;
    };

    const resolveInputEmbeddingStage = (layer, lanes, nowMs = NaN) => {
        if (!layer || layer.index !== 0 || !Array.isArray(lanes) || !lanes.length) return null;
        let hasEmbeddingLanes = false;
        let anyPosWorkRemaining = false;
        let anyTokenPassActive = false;
        let anyPosPassActive = false;
        let anySumActive = false;
        let anyPendingPosPass = false;
        const positionChipRiseStarted = hasGateTokenRiseStarted(pipeline?.__inputPositionChipGate, nowMs);

        for (const lane of lanes) {
            if (!lane || !lane.posVec) continue;
            hasEmbeddingLanes = true;
            if (lane.posAddComplete) continue;
            anyPosWorkRemaining = true;

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

        if (!hasEmbeddingLanes || !anyPosWorkRemaining) return null;
        if (anySumActive) return { eqKey: 'embed_sum', eqTitle: 'Embeddings Computation' };
        if (anyPosPassActive || positionChipRiseStarted) return { eqKey: 'embed_pos', eqTitle: 'Embeddings Computation' };
        if (anyTokenPassActive || anyPendingPosPass) {
            return { eqKey: 'embed_token', eqTitle: 'Embeddings Computation' };
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
        const layerIndex = Number.isFinite(targetLayer?.index) ? Math.floor(targetLayer.index) : null;
        if (Number.isFinite(layerIndex) && Number.isFinite(eqLastLayerIndex) && layerIndex !== eqLastLayerIndex) {
            eqResidualLockLayerIndex = null;
        }
        if (Number.isFinite(layerIndex)) {
            eqLastLayerIndex = layerIndex;
        }
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
        const nowMs = getNowMs();

        // Once Residual Add 2 has been reached for a layer, keep it latched
        // until the pipeline advances to the next layer. This removes
        // transient handoff flicker where the HUD can briefly jump back to
        // another equation between the final residual add and the next layer's LN1.
        if (Number.isFinite(layerIndex) && eqResidualLockLayerIndex === layerIndex) {
            const key = 'resid2';
            const signature = key;
            if (signature === appState.lastEqSignature) return;
            appState.lastEqKey = key;
            appState.lastEqSignature = signature;
            renderEq(EQ[key] || '', 'Residual Add 2');
            return;
        }

        const inputEmbeddingStage = resolveInputEmbeddingStage(targetLayer, lanes, nowMs);
        if (inputEmbeddingStage?.eqKey) {
            const key = inputEmbeddingStage.eqKey;
            const signature = key;
            if (signature === appState.lastEqSignature) return;
            appState.lastEqKey = key;
            appState.lastEqSignature = signature;
            renderEq(EQ[key] || '', inputEmbeddingStage.eqTitle);
            return;
        }

        // Prevent end-of-layer equation flicker:
        // once every lane has reached LN2 done, keep showing Residual Add 2
        // until the pipeline advances to the next layer.
        const allLn2Done = lanes.length > 0 && lanes.every(l => l && l.ln2Phase === 'done');
        const resid2Active = allLn2Done || lanes.some(l => l && l.stopRise && l.ln2Phase === 'done');
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
            if (adding) {
                key = 'resid2';
                title = 'Residual Add 2';
            } else {
                const mlpStage = resolveMlpOverlayStage(lanes);
                key = mlpStage?.eqKey || 'mlp_up';
                title = mlpStage?.eqTitle || 'MLP Up Projection';
            }
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
                title = 'Multi-Head Self-Attention';
            } else if (concatActive) {
                key = 'concat_proj';
                title = 'Concat Heads + W_O';
            } else if (phase === 'parallel_pass_through_active' || phase === 'mha_pass_through_complete') {
                key = 'attn';
                title = 'Multi-Head Self-Attention';
            } else {
                key = 'qkv_per_head';
                title = 'Q/K/V Projections';
            }
        } else if (ln1Active || targetLayer.isActive) {
            key = 'ln1';
            title = 'LayerNorm 1';
        }
        if (!key) return;
        if (key === 'resid2' && Number.isFinite(layerIndex)) {
            eqResidualLockLayerIndex = layerIndex;
        }
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
            eqBody = buildLayerNormEquation(U, X_LN, { norm: normT, scale: scaleT, shift: shiftT });
            signature = `${key}|n${normT.toFixed(3)}|s${scaleT.toFixed(3)}|h${shiftT.toFixed(3)}`;
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
        const eased = getTopEmbeddingActivationEasedProgress(entryY, exitY, highestY);
        // Avoid re-applying the base gray style before activation starts.
        if (eased <= 0 && !appState.topEmbedActivated) {
            return;
        }
        if (eased >= 1) {
            if (appState.topEmbedActivated) return;
            topEmbedWorkingColor.copy(topEmbedTargetColor);
            appState.vocabTopRef.setColor(topEmbedWorkingColor);
            appState.vocabTopRef.setMaterialProperties({
                emissiveIntensity: TOP_EMBED_BASE_EMISSIVE + topEmbedMaxEmissive
            });
            appState.topEmbedActivated = true;
            return;
        }
        topEmbedWorkingColor.copy(topEmbedBaseColor).lerp(topEmbedTargetColor, eased);
        appState.vocabTopRef.setColor(topEmbedWorkingColor);
        appState.vocabTopRef.setMaterialProperties({
            emissiveIntensity: TOP_EMBED_BASE_EMISSIVE + topEmbedMaxEmissive * eased
        });
    }

    let lastStatusText = '';

    function updateStatus() {
        if (!statusDiv) return;
        const nowMs = getNowMs();
        const idx = Number.isFinite(pipeline._currentLayerIdx) ? pipeline._currentLayerIdx : 0;
        const total = NUM_LAYERS;
        const stageOverride = resolveFinalStage();
        const isFinalStage = idx >= total || stageOverride?.active;
        const layer = (isFinalStage ? stageOverride?.layer : null) ?? pipeline._layers[idx];
        let displayStage = 'Loading...';
        let hideLayerHeader = false;

        if (isFinalStage) {
            displayStage = stageOverride?.status || 'Output Logits';
        } else if (layer && Array.isArray(layer.lanes) && layer.lanes.length) {
            const lanes = layer.lanes;
            const inputEmbeddingStage = resolveInputEmbeddingStage(layer, lanes, nowMs);
            if (inputEmbeddingStage?.eqKey) {
                displayStage = 'Embeddings Computation';
                hideLayerHeader = true;
            } else {
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
                    displayStage = 'Multi-Head Self-Attention';
                } else if (layer.isActive) {
                    displayStage = 'LayerNorm 1';
                }
            }
        }

        if (isFinalStage) {
            updateEquations(layer, stageOverride || { layer, eqKey: 'logits', eqTitle: 'Output Logits', status: 'Output Logits', active: true });
        } else {
            updateEquations(layer);
        }

        const showStageLine = Boolean(displayStage);
        const safeIdx = Math.max(0, Math.min(total - 1, idx));
        const headerLine = isFinalStage
            ? 'Output Head'
            : (hideLayerHeader ? '' : `Layer ${safeIdx + 1}`);
        const kvCachePhase = appState.kvCachePrefillActive ? 'prefill' : 'decode';
        const kvCacheStatusText = appState.kvCacheModeEnabled
            ? buildKvCacheOverlayBadgeText(kvCachePhase)
            : '';
        const nextStatusText = showStageLine
            ? (headerLine
                ? `${headerLine}\n${displayStage}`
                : `${displayStage}`)
            : headerLine;
        const nextStatusKey = `${nextStatusText}||${kvCacheStatusText}`;
        if (nextStatusKey === lastStatusText) {
            checkTopEmbeddingActivation();
            return;
        }

        lastStatusText = nextStatusKey;
        if (statusTextEl) {
            renderStatusText({
                headerLine,
                stageLine: showStageLine ? displayStage : ''
            });
        } else {
            statusDiv.textContent = nextStatusText;
        }
        if (statusKvCacheLink) {
            const showKvLink = appState.kvCacheModeEnabled;
            statusKvCacheLink.hidden = !showKvLink;
            statusKvCacheLink.setAttribute('aria-hidden', showKvLink ? 'false' : 'true');
            statusKvCacheLink.textContent = kvCacheStatusText;
            if (showKvLink) {
                statusKvCacheLink.setAttribute('aria-label', `${kvCacheStatusText}. Open KV cache details.`);
            } else {
                statusKvCacheLink.removeAttribute('aria-label');
            }
        }
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
