import { shouldWaitForInputChipGate } from '../engine/layers/gpt2InputChipGateUtils.js';

const LIVE_SCENE_DETAIL_LABELS = Object.freeze({
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
});

function getNowMs() {
    return (
        (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now()
            : Date.now()
    );
}

function pickLane(items, predicate) {
    if (!Array.isArray(items)) return null;
    return items.find(predicate) || null;
}

function hasGateTokenRiseStarted(gate, nowMs) {
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
}

function resolveInputEmbeddingStage(pipeline, layer, lanes, nowMs = NaN) {
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
    if (anySumActive) return { key: 'embed_sum', label: 'Embeddings Computation', detailLabel: 'Embedding sum' };
    if (anyPosPassActive || positionChipRiseStarted) {
        return { key: 'embed_pos', label: 'Embeddings Computation', detailLabel: 'Position embedding' };
    }
    if (anyTokenPassActive || anyPendingPosPass) {
        return { key: 'embed_token', label: 'Embeddings Computation', detailLabel: 'Token embedding' };
    }
    return null;
}

function getTopEmbeddingStats(lastLayer) {
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
}

function resolveFinalStage(pipeline, totalLayers) {
    const layers = Array.isArray(pipeline?._layers) ? pipeline._layers : [];
    const lastLayer = layers[Math.max(0, totalLayers - 1)];
    if (!lastLayer) return null;
    const lanes = Array.isArray(lastLayer.lanes) ? lastLayer.lanes : [];
    const topLnActive = lanes.some((lane) => (
        lane?.__topLnEntered
        || lane?.__topLnMultStarted
        || lane?.__topLnShiftStarted
        || lane?.__topLnShiftComplete
    ));
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
        return {
            layer: lastLayer,
            active: true,
            label: 'Output Logits',
            detailLabel: 'Output logits / unembedding'
        };
    }
    if (topLnActive) {
        return {
            layer: lastLayer,
            active: true,
            label: 'Final LayerNorm',
            detailLabel: 'Final LayerNorm'
        };
    }
    return {
        layer: lastLayer,
        active: false,
        label: 'Final LayerNorm',
        detailLabel: 'Final LayerNorm'
    };
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
        if (tokenInside) return false;
        return chipY >= entryStartY - 0.01 && chipY <= targetY + 0.01;
    });
}

function detectInputVocabPassLane(pipeline, layer) {
    if (!pipeline || layer?.index !== 0) return null;

    const gate = pipeline.__inputVocabChipGate;
    const nowMs = getNowMs();

    return pickLane(layer?.lanes || [], (lane) => {
        const currentY = lane?.originalVec?.group?.position?.y;
        const travelStartY = Number.isFinite(lane?.vocabEmbeddingTravelStartY)
            ? lane.vocabEmbeddingTravelStartY
            : NaN;
        const exitY = Number.isFinite(lane?.vocabEmbeddingExitY)
            ? lane.vocabEmbeddingExitY
            : NaN;

        if (!Number.isFinite(currentY) || !Number.isFinite(travelStartY) || !Number.isFinite(exitY)) {
            return false;
        }

        if (currentY <= travelStartY + 0.01 || currentY >= exitY - 0.01) {
            return false;
        }

        return !shouldWaitForInputChipGate(gate, lane.tokenIndex, nowMs);
    });
}

function detectPreciseStage(pipeline, totalLayers) {
    const layers = pipeline?._layers;
    if (!Array.isArray(layers) || !layers.length) return null;

    const currentIdx = Number.isFinite(pipeline._currentLayerIdx) ? pipeline._currentLayerIdx : 0;
    if (currentIdx >= totalLayers) {
        const lastLayer = layers[Math.max(0, totalLayers - 1)];
        const lanes = lastLayer?.lanes || [];
        const shiftLane = pickLane(lanes, (lane) => lane.__topLnShiftStarted && !lane.__topLnShiftComplete);
        if (shiftLane) {
            return { stage: 'final_ln_shift', layer: null, lane: shiftLane };
        }
        const scaleLane = pickLane(lanes, (lane) => lane.__topLnMultStarted && !lane.__topLnShiftStarted);
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
            (lane) => lane.posVec && !lane.posAddComplete && (lane.__posPassStarted || lane.posAddStarted)
        );
        if (posLane) {
            return { stage: 'position_embedding', layer: null, lane: posLane };
        }
    }

    const ln1ShiftLane = pickLane(lanes, (lane) => lane.ln1AddStarted && !lane.ln1AddComplete);
    if (ln1ShiftLane) {
        return { stage: 'ln1_shift', layer: layer.index, lane: ln1ShiftLane };
    }
    const ln1ScaleLane = pickLane(lanes, (lane) => lane.multStarted && !lane.ln1AddStarted);
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

    const ln2ShiftLane = pickLane(lanes, (lane) => lane.ln2AddStarted && !lane.ln2AddComplete);
    if (ln2ShiftLane) {
        return { stage: 'ln2_shift', layer: layer.index, lane: ln2ShiftLane };
    }
    const ln2ScaleLane = pickLane(lanes, (lane) => lane.multDoneLN2 && !lane.ln2AddStarted);
    if (ln2ScaleLane) {
        return { stage: 'ln2_scale', layer: layer.index, lane: ln2ScaleLane };
    }

    const mlpUpLane = pickLane(lanes, (lane) => lane.mlpUpStarted && !lane.expandedVecGroup);
    if (mlpUpLane) {
        return { stage: 'mlp_up_projection', layer: layer.index, lane: mlpUpLane };
    }

    const mlpDownLane = pickLane(lanes, (lane) => lane.mlpDownStarted && !lane.finalVecAfterMlp);
    if (mlpDownLane) {
        return { stage: 'mlp_down_projection', layer: layer.index, lane: mlpDownLane };
    }

    return null;
}

export function formatLiveSceneDetailLabel(stage = '') {
    return LIVE_SCENE_DETAIL_LABELS[stage] || '';
}

export function resolveLiveSceneState({
    pipeline = null,
    totalLayers = null
} = {}) {
    if (!pipeline) return null;
    const layers = Array.isArray(pipeline._layers) ? pipeline._layers : [];
    const resolvedTotalLayers = Number.isFinite(totalLayers)
        ? Math.max(1, Math.floor(totalLayers))
        : (Number.isFinite(pipeline._numLayers) ? Math.max(1, Math.floor(pipeline._numLayers)) : Math.max(1, layers.length));
    const currentIdx = Number.isFinite(pipeline._currentLayerIdx) ? pipeline._currentLayerIdx : 0;
    const nowMs = getNowMs();
    const finalStage = resolveFinalStage(pipeline, resolvedTotalLayers);
    const isFinalStage = currentIdx >= resolvedTotalLayers || finalStage?.active;
    const layer = (isFinalStage ? finalStage?.layer : null) ?? layers[Math.min(layers.length - 1, Math.max(0, currentIdx))];
    const layerIndex = Number.isFinite(layer?.index)
        ? Math.floor(layer.index)
        : (isFinalStage ? Math.max(0, resolvedTotalLayers - 1) : Math.max(0, Math.min(resolvedTotalLayers - 1, currentIdx)));
    let displayStage = '';
    let detailLabel = '';

    if (isFinalStage) {
        displayStage = finalStage?.label || 'Output Logits';
        detailLabel = finalStage?.detailLabel || displayStage;
    } else if (layer && Array.isArray(layer.lanes) && layer.lanes.length) {
        const lanes = layer.lanes;
        const inputEmbeddingStage = resolveInputEmbeddingStage(pipeline, layer, lanes, nowMs);
        if (inputEmbeddingStage?.label) {
            displayStage = inputEmbeddingStage.label;
            detailLabel = inputEmbeddingStage.detailLabel || '';
        } else {
            const mlpActive = lanes.some((lane) => lane.mlpUpStarted || lane.ln2Phase === 'mlpReady' || lane.ln2Phase === 'done');
            const ln2Active = !mlpActive && lanes.some((lane) => lane.ln2Phase && lane.ln2Phase !== 'notStarted');
            const ln1Active = !mlpActive && !ln2Active && lanes.some((lane) => ['waiting', 'right', 'insideLN'].includes(lane.horizPhase));
            const mhsaActive = !mlpActive && !ln2Active && !ln1Active && (
                layer._mhsaStart === true
                || lanes.some((lane) => ['riseAboveLN', 'readyMHSA', 'travelMHSA', 'postMHSAAddition', 'waitingForLN2'].includes(lane.horizPhase))
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

    const preciseStage = detectPreciseStage(pipeline, resolvedTotalLayers);
    const preciseStageLabel = preciseStage?.stage ? formatLiveSceneDetailLabel(preciseStage.stage) : '';
    if (!detailLabel && preciseStageLabel) {
        detailLabel = preciseStageLabel;
    }

    return {
        totalLayers: resolvedTotalLayers,
        currentLayerIndex: Number.isFinite(layerIndex) ? layerIndex : null,
        displayStage,
        detailStageKey: preciseStage?.stage || '',
        detailStageLabel: detailLabel,
        detailStageLayerIndex: Number.isFinite(preciseStage?.layer) ? preciseStage.layer : null,
        isFinalStage,
        forwardPassComplete: typeof pipeline?.isForwardPassComplete === 'function'
            ? pipeline.isForwardPassComplete()
            : false
    };
}
