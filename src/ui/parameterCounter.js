import { PARAMETER_CHECKPOINTS } from '../data/parameterCheckpoints.js';
import { appState } from '../state/appState.js';
import {
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

function estimateStageDuration(stage, layer, lane) {
    const speedScale = BASE_SPEED_MULT / Math.max(1, GLOBAL_ANIM_SPEED_MULT);
    switch (stage) {
        case 'token_embedding':
            return 700 * speedScale;
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

    const seedTokenEmbedding = () => {
        const key = stageKey('token_embedding', null);
        const idx = indexByKey.get(key);
        if (idx == null) return;
        const entry = checkpoints[idx];
        queue.push({
            entry,
            durationMs: estimateStageDuration('token_embedding', null, null),
            label: formatStageLabel('token_embedding', null),
            minDuration: MIN_STAGE_MS,
            index: idx,
        });
        lastIndex = idx;
        startNext();
    };

    const resetCounterForNewPass = () => {
        currentValue = 0;
        lastIndex = -1;
        activeAnim = null;
        pauseStamp = null;
        queue.length = 0;
        counter.dataset.animating = 'false';
        if (stageEl) stageEl.textContent = '';
        renderValue(currentValue);
        seedTokenEmbedding();
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
