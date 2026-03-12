import { PARAMETER_CHECKPOINTS } from '../data/parameterCheckpoints.js';
import { appState } from '../state/appState.js';
import {
    ANIM_RISE_SPEED_ORIGINAL,
    ANIM_RISE_SPEED_INSIDE_LN,
    GLOBAL_ANIM_SPEED_MULT,
    MHSA_PASS_THROUGH_TOTAL_DURATION_MS,
    MLP_MATRIX_PARAMS_DOWN,
    MLP_MATRIX_PARAMS_UP,
    OUTPUT_PROJ_STAGE2_MS,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_SPEED_MULT,
    VECTOR_LENGTH_PRISM
} from '../utils/constants.js';
import {
    MHA_FINAL_K_COLOR,
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MLP_DOWN_MATRIX_COLOR,
    MLP_UP_MATRIX_COLOR,
    POSITION_EMBED_COLOR
} from '../animations/LayerAnimationConstants.js';
import { shouldWaitForInputChipGate } from '../engine/layers/gpt2InputChipGateUtils.js';
import {
    INPUT_VOCAB_RISE_SPEED_AT_REVEAL,
    INPUT_VOCAB_RISE_SPEED_NEAR_EXIT,
    INPUT_VOCAB_RISE_SPEED_PRE_REVEAL
} from '../engine/layers/gpt2InputVocabTravelUtils.js';

const STAGE_LABELS = {
    token_embedding: 'Token embedding',
    position_embedding: 'Position embedding',
    ln1_scale: 'LayerNorm 1 scale',
    ln1_shift: 'LayerNorm 1 shift',
    qkv_projection: 'Q/K/V projections',
    attn_output_projection: 'Attention output projection',
    ln2_scale: 'LayerNorm 2 scale',
    ln2_shift: 'LayerNorm 2 shift',
    mlp_up_projection: 'MLP up projection',
    mlp_down_projection: 'MLP down projection',
    final_ln_scale: 'Final LayerNorm scale',
    final_ln_shift: 'Final LayerNorm shift'
};

const BASE_SPEED_MULT = 100;
const MIN_STAGE_MS = 280;
const DEFAULT_STAGE_MS = 900;
const SKIP_MIN_STAGE_MS = 120;
const COUNTER_DEFAULT_COLOR = 0xf3f7fb;
const COUNTER_LAYER_NORM_COLOR = 0xffffff;

const COUNTER_STAGE_PALETTES = Object.freeze({
    token_embedding: [MHA_FINAL_Q_COLOR],
    position_embedding: [POSITION_EMBED_COLOR],
    ln1_scale: [COUNTER_LAYER_NORM_COLOR],
    ln1_shift: [COUNTER_LAYER_NORM_COLOR],
    qkv_projection: [MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR, MHA_FINAL_V_COLOR],
    attn_output_projection: [MHA_OUTPUT_PROJECTION_MATRIX_COLOR],
    ln2_scale: [COUNTER_LAYER_NORM_COLOR],
    ln2_shift: [COUNTER_LAYER_NORM_COLOR],
    mlp_up_projection: [MLP_UP_MATRIX_COLOR],
    mlp_down_projection: [MLP_DOWN_MATRIX_COLOR],
    final_ln_scale: [COUNTER_LAYER_NORM_COLOR],
    final_ln_shift: [COUNTER_LAYER_NORM_COLOR]
});

const COUNTER_STAGE_GRADIENT_MODES = Object.freeze({
    qkv_projection: 'bands'
});

function clampChannel(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(255, Math.round(value)));
}

function normalizeHexColor(value) {
    const safe = Number.isFinite(value) ? value : COUNTER_DEFAULT_COLOR;
    return safe & 0xffffff;
}

function colorToRgb(hexColor) {
    const safe = normalizeHexColor(hexColor);
    return {
        r: clampChannel((safe >> 16) & 0xff),
        g: clampChannel((safe >> 8) & 0xff),
        b: clampChannel(safe & 0xff)
    };
}

function colorToCss(hexColor) {
    const safe = normalizeHexColor(hexColor);
    return `#${safe.toString(16).padStart(6, '0')}`;
}

function formatGlowColor(hexColor, alpha = 1) {
    const { r, g, b } = colorToRgb(hexColor);
    const safeAlpha = Number.isFinite(alpha) ? Math.max(0, Math.min(1, alpha)) : 1;
    return `rgba(${r}, ${g}, ${b}, ${safeAlpha})`;
}

function clamp01(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(1, value));
}

function buildGradientCss(palette, mode = 'smooth') {
    const colors = Array.isArray(palette) && palette.length
        ? palette
        : [COUNTER_DEFAULT_COLOR];

    if (colors.length === 1) {
        const cssColor = colorToCss(colors[0]);
        return `linear-gradient(90deg, ${cssColor} 0%, ${cssColor} 100%)`;
    }

    if (mode === 'bands') {
        const segmentWidth = 100 / colors.length;
        const stops = [];
        for (let i = 0; i < colors.length; i++) {
            const start = i * segmentWidth;
            const end = (i + 1) * segmentWidth;
            const cssColor = colorToCss(colors[i]);
            stops.push(`${cssColor} ${start.toFixed(1)}%`);
            stops.push(`${cssColor} ${end.toFixed(1)}%`);
        }
        return `linear-gradient(90deg, ${stops.join(', ')})`;
    }

    const lastIndex = colors.length - 1;
    const stops = colors.map((color, index) => {
        const pct = lastIndex > 0 ? (index / lastIndex) * 100 : 0;
        return `${colorToCss(color)} ${pct.toFixed(1)}%`;
    });
    return `linear-gradient(90deg, ${stops.join(', ')})`;
}

function applyStageCounterGlow(counter, stage) {
    if (!counter) return;

    const palette = COUNTER_STAGE_PALETTES[stage] || [COUNTER_DEFAULT_COLOR];
    const gradientMode = COUNTER_STAGE_GRADIENT_MODES[stage] || 'smooth';
    const midIndex = Math.floor(palette.length / 2);
    const fillColor = palette[midIndex] ?? COUNTER_DEFAULT_COLOR;
    const nearColor = palette[0] ?? fillColor;
    const midColor = palette[midIndex] ?? fillColor;
    const farColor = palette[palette.length - 1] ?? fillColor;

    counter.style.setProperty('--parameter-counter-value-color', colorToCss(fillColor));
    counter.style.setProperty('--parameter-counter-value-gradient', buildGradientCss(palette, gradientMode));
    counter.style.setProperty('--parameter-counter-glow-near', formatGlowColor(nearColor, 0.28));
    counter.style.setProperty('--parameter-counter-glow-mid', formatGlowColor(midColor, 0.14));
    counter.style.setProperty('--parameter-counter-glow-far', formatGlowColor(farColor, 0.06));
}

function formatMillions(value) {
    const millions = value / 1_000_000;
    return `${millions.toFixed(2)} M`;
}

function stageKey(stage, layer) {
    return `${stage}|${layer == null ? 'global' : layer}`;
}

function estimateAdditionDuration(vec) {
    const length = Math.max(1, Math.floor(vec?.instanceCount || VECTOR_LENGTH_PRISM));
    const total = PRISM_ADD_ANIM_BASE_DURATION
        + PRISM_ADD_ANIM_BASE_FLASH_DURATION
        + length * PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS;
    return total / Math.max(1, PRISM_ADD_ANIM_SPEED_MULT);
}

function estimateMlpUpDuration(layer, lane) {
    if (!layer?.mlpUp?.group || !lane?.resultVecLN2?.group) return DEFAULT_STAGE_MS;
    const topY = layer.mlpUp.group.position.y + MLP_MATRIX_PARAMS_UP.height / 2;
    const distance = topY - lane.resultVecLN2.group.position.y;
    const speed = ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT;
    if (!Number.isFinite(distance) || !Number.isFinite(speed) || speed <= 0) return DEFAULT_STAGE_MS;
    return Math.max(MIN_STAGE_MS, Math.abs(distance) / speed * 1000);
}

function estimateMlpDownDuration(layer, lane) {
    if (!layer?.mlpDown?.group || !lane?.expandedVecGroup?.position) return DEFAULT_STAGE_MS;
    const topY = layer.mlpDown.group.position.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
    const distance = topY - lane.expandedVecGroup.position.y;
    const speed = ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT;
    if (!Number.isFinite(distance) || !Number.isFinite(speed) || speed <= 0) return DEFAULT_STAGE_MS;
    return Math.max(MIN_STAGE_MS, Math.abs(distance) / speed * 1000);
}

function invertQuadraticOut(progress) {
    const clamped = clamp01(progress);
    return 1 - Math.sqrt(Math.max(0, 1 - clamped));
}

function estimateTokenEmbeddingChipDuration(entry) {
    const durationMs = Number.isFinite(entry?.durationMs) ? entry.durationMs : NaN;
    const startY = Number.isFinite(entry?.startY) ? entry.startY : NaN;
    const targetY = Number.isFinite(entry?.targetY) ? entry.targetY : NaN;
    const currentY = Number.isFinite(entry?.chip?.position?.y)
        ? entry.chip.position.y
        : (Number.isFinite(entry?.entryStartY) ? entry.entryStartY : NaN);
    const totalDistance = targetY - startY;

    if (
        !Number.isFinite(durationMs)
        || durationMs <= 0
        || !Number.isFinite(startY)
        || !Number.isFinite(targetY)
        || !Number.isFinite(currentY)
        || !Number.isFinite(totalDistance)
        || totalDistance <= 1e-6
    ) {
        return NaN;
    }

    const currentProgress = clamp01((currentY - startY) / totalDistance);
    const currentT = invertQuadraticOut(currentProgress);
    return Math.max(MIN_STAGE_MS, durationMs * Math.max(0, 1 - currentT));
}

function estimateTokenEmbeddingDuration(lane) {
    const chipDuration = estimateTokenEmbeddingChipDuration(lane);
    if (Number.isFinite(chipDuration)) {
        return chipDuration;
    }

    const speed = ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT;
    const fallbackMs = 700 * (BASE_SPEED_MULT / Math.max(1, GLOBAL_ANIM_SPEED_MULT));
    const travelStartY = Number.isFinite(lane?.vocabEmbeddingTravelStartY)
        ? lane.vocabEmbeddingTravelStartY
        : NaN;
    const exitY = Number.isFinite(lane?.vocabEmbeddingExitY)
        ? lane.vocabEmbeddingExitY
        : NaN;

    if (!Number.isFinite(speed) || speed <= 0 || !Number.isFinite(travelStartY) || !Number.isFinite(exitY) || exitY <= travelStartY) {
        return fallbackMs;
    }

    const revealYRaw = Number.isFinite(lane?.vocabEmbeddingRevealY)
        ? lane.vocabEmbeddingRevealY
        : NaN;
    const revealY = Number.isFinite(revealYRaw)
        ? Math.max(travelStartY, Math.min(exitY, revealYRaw))
        : NaN;

    if (!Number.isFinite(revealY) || revealY <= travelStartY + 0.01 || revealY >= exitY - 0.01) {
        const averageRiseMult = (INPUT_VOCAB_RISE_SPEED_PRE_REVEAL + INPUT_VOCAB_RISE_SPEED_NEAR_EXIT) / 2;
        return Math.max(MIN_STAGE_MS, ((exitY - travelStartY) / (speed * averageRiseMult)) * 1000);
    }

    const preRevealDistance = Math.max(0, revealY - travelStartY);
    const postRevealDistance = Math.max(0, exitY - revealY);
    const preRevealSpeedMult = (INPUT_VOCAB_RISE_SPEED_PRE_REVEAL + INPUT_VOCAB_RISE_SPEED_AT_REVEAL) / 2;
    const postRevealSpeedMult = (INPUT_VOCAB_RISE_SPEED_AT_REVEAL + INPUT_VOCAB_RISE_SPEED_NEAR_EXIT) / 2;
    const preRevealMs = preRevealDistance / (speed * preRevealSpeedMult) * 1000;
    const postRevealMs = postRevealDistance / (speed * postRevealSpeedMult) * 1000;

    return Math.max(MIN_STAGE_MS, preRevealMs + postRevealMs);
}

function estimateStageDuration(stage, layer, lane) {
    const speedScale = BASE_SPEED_MULT / Math.max(1, GLOBAL_ANIM_SPEED_MULT);
    switch (stage) {
        case 'token_embedding':
            return estimateTokenEmbeddingDuration(lane);
        case 'position_embedding':
            return 1200 * speedScale;
        case 'ln1_scale':
        case 'ln2_scale':
        case 'final_ln_scale':
            return 420 * speedScale;
        case 'ln1_shift':
        case 'ln2_shift':
        case 'final_ln_shift':
            return estimateAdditionDuration(
                lane?.resultVec || lane?.resultVecLN2 || lane?.originalVec || lane?.addTarget || lane?.addTargetLN2
            );
        case 'qkv_projection':
            return MHSA_PASS_THROUGH_TOTAL_DURATION_MS / Math.max(1, GLOBAL_ANIM_SPEED_MULT);
        case 'attn_output_projection':
            return OUTPUT_PROJ_STAGE2_MS / Math.max(1, GLOBAL_ANIM_SPEED_MULT);
        case 'mlp_up_projection':
            return estimateMlpUpDuration(layer, lane);
        case 'mlp_down_projection':
            return estimateMlpDownDuration(layer, lane);
        default:
            return DEFAULT_STAGE_MS * speedScale;
    }
}

function formatStageLabel(stage, layer) {
    const label = STAGE_LABELS[stage] || stage;
    if (typeof layer === 'number') {
        return `Layer ${layer + 1}: ${label}`;
    }
    return label;
}

function isPaused(pipeline) {
    if (pipeline?.engine && pipeline.engine._paused) return true;
    if (appState?.userPaused || appState?.modalPaused) return true;
    return false;
}

function pickLane(lanes, predicate) {
    if (!Array.isArray(lanes)) return null;
    return lanes.find(predicate) || null;
}

function detectInputVocabChipEntry(pipeline) {
    const gate = pipeline?.__inputVocabChipGate;
    const entries = Array.isArray(gate?.chipEntries) ? gate.chipEntries : [];
    if (!entries.length) return null;

    return pickLane(entries, (entry) => {
        const chipY = entry?.chip?.position?.y;
        const entryStartY = Number.isFinite(entry?.entryStartY) ? entry.entryStartY : NaN;
        const targetY = Number.isFinite(entry?.targetY) ? entry.targetY : NaN;
        const tokenKey = typeof entry?.tokenKey === 'string' ? entry.tokenKey : null;
        const tokenInside = !!(
            tokenKey
            && gate?.insideByToken
            && Object.prototype.hasOwnProperty.call(gate.insideByToken, tokenKey)
            && gate.insideByToken[tokenKey] === true
        );

        if (!Number.isFinite(chipY) || !Number.isFinite(entryStartY) || !Number.isFinite(targetY)) {
            return false;
        }
        if (tokenInside) {
            return false;
        }
        return chipY >= entryStartY - 0.01 && chipY <= targetY + 0.01;
    });
}

function detectInputVocabPassLane(pipeline, layer) {
    if (!pipeline || layer?.index !== 0) return null;

    const gate = pipeline.__inputVocabChipGate;
    const nowMs = (typeof performance !== 'undefined' && typeof performance.now === 'function')
        ? performance.now()
        : Date.now();

    return pickLane(layer?.lanes || [], (lane) => {
        const currentY = lane?.originalVec?.group?.position?.y;
        const travelStartY = Number.isFinite(lane?.vocabEmbeddingTravelStartY)
            ? lane.vocabEmbeddingTravelStartY
            : NaN;
        const exitY = Number.isFinite(lane?.vocabEmbeddingExitY)
            ? lane.vocabEmbeddingExitY
            : NaN;

        if (
            !Number.isFinite(currentY)
            || !Number.isFinite(travelStartY)
            || !Number.isFinite(exitY)
        ) {
            return false;
        }

        if (currentY <= travelStartY + 0.01 || currentY >= exitY - 0.01) {
            return false;
        }

        return !shouldWaitForInputChipGate(gate, lane.tokenIndex, nowMs);
    });
}

function detectStage(pipeline, numLayers) {
    const layers = pipeline?._layers;
    if (!Array.isArray(layers) || !layers.length) return null;

    const currentIdx = Number.isFinite(pipeline._currentLayerIdx) ? pipeline._currentLayerIdx : 0;
    if (currentIdx >= numLayers) {
        const lastLayer = layers[numLayers - 1];
        const lanes = lastLayer?.lanes || [];
        const shiftLane = pickLane(lanes, (l) => l.__topLnShiftStarted && !l.__topLnShiftComplete);
        if (shiftLane) {
            return { stage: 'final_ln_shift', layer: null, lane: shiftLane };
        }
        const scaleLane = pickLane(lanes, (l) => l.__topLnMultStarted && !l.__topLnShiftStarted);
        if (scaleLane) {
            return { stage: 'final_ln_scale', layer: null, lane: scaleLane };
        }
        return null;
    }

    const layer = layers[Math.min(layers.length - 1, Math.max(0, currentIdx))];
    if (!layer) return null;

    const lanes = layer.lanes || [];
    const inputVocabChipEntry = detectInputVocabChipEntry(pipeline);
    if (inputVocabChipEntry) {
        return { stage: 'token_embedding', layer: null, lane: inputVocabChipEntry };
    }
    const inputVocabLane = detectInputVocabPassLane(pipeline, layer);
    if (inputVocabLane) {
        return { stage: 'token_embedding', layer: null, lane: inputVocabLane };
    }

    if (layer.index === 0) {
        const posLane = pickLane(
            lanes,
            (l) => l.posVec && !l.posAddComplete && (l.__posPassStarted || l.posAddStarted)
        );
        if (posLane) {
            return { stage: 'position_embedding', layer: null, lane: posLane };
        }
    }

    const ln1ShiftLane = pickLane(lanes, (l) => l.ln1AddStarted && !l.ln1AddComplete);
    if (ln1ShiftLane) {
        return { stage: 'ln1_shift', layer: layer.index, lane: ln1ShiftLane };
    }
    const ln1ScaleLane = pickLane(lanes, (l) => l.multStarted && !l.ln1AddStarted);
    if (ln1ScaleLane) {
        return { stage: 'ln1_scale', layer: layer.index, lane: ln1ScaleLane };
    }

    const mhsa = layer.mhsaAnimation;
    if (mhsa && mhsa.outputProjMatrixAnimationPhase === 'vectors_inside') {
        return { stage: 'attn_output_projection', layer: layer.index, lane: lanes[0] || null };
    }
    if (mhsa && mhsa.mhaPassThroughPhase === 'parallel_pass_through_active') {
        return { stage: 'qkv_projection', layer: layer.index, lane: lanes[0] || null };
    }

    const ln2ShiftLane = pickLane(lanes, (l) => l.ln2AddStarted && !l.ln2AddComplete);
    if (ln2ShiftLane) {
        return { stage: 'ln2_shift', layer: layer.index, lane: ln2ShiftLane };
    }
    const ln2ScaleLane = pickLane(lanes, (l) => l.multDoneLN2 && !l.ln2AddStarted);
    if (ln2ScaleLane) {
        return { stage: 'ln2_scale', layer: layer.index, lane: ln2ScaleLane };
    }

    const mlpUpLane = pickLane(lanes, (l) => l.mlpUpStarted && !l.expandedVecGroup);
    if (mlpUpLane) {
        return { stage: 'mlp_up_projection', layer: layer.index, lane: mlpUpLane };
    }

    const mlpDownLane = pickLane(lanes, (l) => l.mlpDownStarted && !l.finalVecAfterMlp);
    if (mlpDownLane) {
        return { stage: 'mlp_down_projection', layer: layer.index, lane: mlpDownLane };
    }

    return null;
}

export function initParameterCounter(pipeline, numLayers) {
    const counter = document.getElementById('parameterCounter');
    if (!counter) return null;
    const valueEl = document.getElementById('paramValue');
    const stageEl = document.getElementById('paramStage');
    if (!valueEl) return null;

    const checkpoints = Array.isArray(PARAMETER_CHECKPOINTS) ? PARAMETER_CHECKPOINTS : [];
    const indexByKey = new Map();
    checkpoints.forEach((entry, idx) => {
        indexByKey.set(stageKey(entry.stage, entry.layer ?? null), idx);
    });

    let currentValue = 0;
    let lastIndex = -1;
    let activeAnim = null;
    let pauseStamp = null;
    const queue = [];

    const renderValue = (value) => {
        valueEl.textContent = formatMillions(value);
    };

    const startAnimation = (entry, durationMs, label, minDuration = MIN_STAGE_MS, index = null) => {
        const safeDuration = Math.max(minDuration, Math.floor(durationMs || DEFAULT_STAGE_MS));
        activeAnim = {
            start: performance.now(),
            duration: safeDuration,
            from: currentValue,
            to: entry.cumulative,
            index: typeof index === 'number' ? index : null,
        };
        applyStageCounterGlow(counter, entry.stage);
        if (stageEl) {
            stageEl.textContent = label || formatStageLabel(entry.stage, entry.layer ?? null);
        }
        counter.dataset.animating = 'true';
    };

    const startNext = () => {
        if (activeAnim || queue.length === 0) return;
        const next = queue.shift();
        startAnimation(next.entry, next.durationMs, next.label, next.minDuration, next.index);
    };

    const queueStage = (stage, layer, lane) => {
        const key = stageKey(stage, layer);
        const idx = indexByKey.get(key);
        if (idx == null || idx <= lastIndex) return;
        const entry = checkpoints[idx];
        lastIndex = idx;
        const durationMs = estimateStageDuration(stage, layer, lane);
        queue.push({
            entry,
            durationMs,
            label: formatStageLabel(stage, layer),
            minDuration: MIN_STAGE_MS,
            index: idx,
        });
        startNext();
    };

    const resetCounterForNewPass = () => {
        currentValue = 0;
        lastIndex = -1;
        activeAnim = null;
        pauseStamp = null;
        queue.length = 0;
        counter.dataset.animating = 'false';
        applyStageCounterGlow(counter, null);
        if (stageEl) stageEl.textContent = '';
        renderValue(currentValue);
    };

    resetCounterForNewPass();

    if (pipeline && typeof pipeline.addEventListener === 'function') {
        pipeline.addEventListener('passreset', () => {
            resetCounterForNewPass();
        });
    }

    const tick = (now) => {
        const skipping = typeof pipeline?.isSkipToEndActive === 'function' && pipeline.isSkipToEndActive();

        if (isPaused(pipeline)) {
            if (pauseStamp == null) pauseStamp = now;
            requestAnimationFrame(tick);
            return;
        }

        if (pauseStamp != null && activeAnim) {
            activeAnim.start += now - pauseStamp;
            pauseStamp = null;
        } else if (pauseStamp != null) {
            pauseStamp = null;
        }

        const detected = detectStage(pipeline, numLayers);
        if (detected) {
            if (skipping) {
                const key = stageKey(detected.stage, detected.layer);
                const idx = indexByKey.get(key);
                if (idx != null && idx >= lastIndex) {
                    const entry = checkpoints[idx];
                    lastIndex = idx;
                    const durationMs = estimateStageDuration(detected.stage, detected.layer, detected.lane);
                    queue.length = 0;
                    if (!activeAnim || activeAnim.index !== idx) {
                        startAnimation(
                            entry,
                            durationMs,
                            formatStageLabel(detected.stage, detected.layer),
                            SKIP_MIN_STAGE_MS,
                            idx
                        );
                    }
                }
            } else {
                queueStage(detected.stage, detected.layer, detected.lane);
            }
        }

        if (activeAnim) {
            const t = Math.min(1, (now - activeAnim.start) / activeAnim.duration);
            const eased = 1 - Math.pow(1 - t, 3);
            const value = activeAnim.from + (activeAnim.to - activeAnim.from) * eased;
            renderValue(value);
            if (t >= 1) {
                currentValue = activeAnim.to;
                activeAnim = null;
                counter.dataset.animating = 'false';
                startNext();
            }
        } else {
            renderValue(currentValue);
        }

        requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);

    return {
        getCurrentValue: () => currentValue,
    };
}

export default initParameterCounter;
