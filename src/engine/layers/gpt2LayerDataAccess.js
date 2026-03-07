import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { getLayerNormParamData } from '../../data/layerNormParams.js';
import { applyVectorData, formatTokenLabel } from './gpt2LayerUtils.js';
import { formatLayerNormParamLabel } from '../../utils/layerNormLabels.js';

export function resolveBaseVectorLength(layer, fallbackLength = 1) {
    const fallback = Number.isFinite(fallbackLength) ? Math.max(1, Math.floor(fallbackLength)) : 1;
    if (Number.isFinite(layer?._baseVectorLength)) {
        return Math.max(1, Math.floor(layer._baseVectorLength));
    }
    return fallback;
}

export function resolveLaneLayoutCount(layer) {
    const laneCount = Number.isFinite(layer?._laneCount)
        ? Math.max(1, Math.floor(layer._laneCount))
        : 1;
    const configured = Number.isFinite(layer?._laneLayoutCount)
        ? Math.floor(layer._laneLayoutCount)
        : laneCount;
    return Math.max(laneCount, configured);
}

export function resolveActiveLaneLayoutIndices(layer) {
    const laneCount = Number.isFinite(layer?._laneCount)
        ? Math.max(1, Math.floor(layer._laneCount))
        : 1;
    if (Array.isArray(layer?._activeLaneLayoutIndices) && layer._activeLaneLayoutIndices.length) {
        return layer._activeLaneLayoutIndices.slice(0, laneCount);
    }
    return Array.from({ length: laneCount }, (_, idx) => idx);
}

export function resolveInstanceCountFromData(layer, values, fallback = null, fallbackLength = 1) {
    if (Array.isArray(values) || ArrayBuffer.isView(values)) {
        return Math.max(1, values.length || 1);
    }
    const base = Number.isFinite(fallback)
        ? fallback
        : resolveBaseVectorLength(layer, fallbackLength);
    return Math.max(1, Math.floor(base));
}

export function createPrismVectorForLayer(layer, values, position, numSubsections = 30, instanceCount = null, fallbackLength = 1) {
    const count = Number.isFinite(instanceCount)
        ? Math.max(1, Math.floor(instanceCount))
        : resolveInstanceCountFromData(layer, values, null, fallbackLength);
    const data = Array.isArray(values)
        ? values
        : ArrayBuffer.isView(values)
            ? Array.from(values)
            : null;
    return new VectorVisualizationInstancedPrism(data, position, numSubsections, count);
}

export function resolveLayerNormParamDataForLayer(layer, kind, param, fallbackLength = 1) {
    const baseLength = resolveBaseVectorLength(layer, fallbackLength);
    return getLayerNormParamData(layer?.index, kind, param, baseLength);
}

export function applyLayerNormParamVectorForLayer(layer, targetVec, kind, param, colorOptions = null, fallbackLength = 1) {
    if (!targetVec) return false;
    const data = resolveLayerNormParamDataForLayer(layer, kind, param, fallbackLength);
    if (!data) return false;
    const label = formatLayerNormParamLabel(kind, param);
    const meta = {
        stage: `${kind}.param.${param}`,
        layerIndex: layer?.index,
        notes: param === 'scale'
            ? 'LayerNorm scale parameter'
            : 'LayerNorm shift parameter'
    };
    return applyVectorData(targetVec, data, label, meta, colorOptions);
}

export function resolveTokenIndexForLane(layer, laneIdx, laneLayoutIdx = null) {
    if (Array.isArray(layer?._passLaneTokenIndices) && Number.isFinite(layer._passLaneTokenIndices[laneIdx])) {
        return layer._passLaneTokenIndices[laneIdx];
    }
    const resolvedLaneIdx = Number.isFinite(laneLayoutIdx)
        ? Math.floor(laneLayoutIdx)
        : laneIdx;
    if (!layer?.activationSource) return resolvedLaneIdx;
    return layer.activationSource.getLaneTokenIndex(resolvedLaneIdx, resolveLaneLayoutCount(layer));
}

export function resolveTokenLabelForLayer(layer, tokenIndex) {
    if (!layer?.activationSource) return null;
    return formatTokenLabel(layer.activationSource.getTokenString(tokenIndex));
}

function getActivationData(layer, lane, getterName, targetLength) {
    if (!layer?.activationSource || !lane || typeof layer.activationSource[getterName] !== 'function') return null;
    return layer.activationSource[getterName](layer.index, lane.tokenIndex, targetLength);
}

export function getEmbeddingDataForLane(layer, lane, kind, fallbackLength = 1) {
    if (!layer?.activationSource || !lane || typeof layer.activationSource.getEmbedding !== 'function') return null;
    return layer.activationSource.getEmbedding(kind, lane.tokenIndex, resolveBaseVectorLength(layer, fallbackLength));
}

export function getLayerIncomingDataForLane(layer, lane, fallbackLength = 1) {
    return getActivationData(layer, lane, 'getLayerIncoming', resolveBaseVectorLength(layer, fallbackLength));
}

export function getLn1DataForLane(layer, lane, stage, fallbackLength = 1) {
    if (!layer?.activationSource || !lane || typeof layer.activationSource.getLayerLn1 !== 'function') return null;
    return layer.activationSource.getLayerLn1(layer.index, stage, lane.tokenIndex, resolveBaseVectorLength(layer, fallbackLength));
}

export function getLn2DataForLane(layer, lane, stage, fallbackLength = 1) {
    if (!layer?.activationSource || !lane || typeof layer.activationSource.getLayerLn2 !== 'function') return null;
    return layer.activationSource.getLayerLn2(layer.index, stage, lane.tokenIndex, resolveBaseVectorLength(layer, fallbackLength));
}

export function getAttentionOutputProjectionDataForLane(layer, lane, fallbackLength = 1) {
    return getActivationData(layer, lane, 'getAttentionOutputProjection', resolveBaseVectorLength(layer, fallbackLength));
}

export function getPostAttentionResidualDataForLane(layer, lane, fallbackLength = 1) {
    return getActivationData(layer, lane, 'getPostAttentionResidual', resolveBaseVectorLength(layer, fallbackLength));
}

export function getMlpUpDataForLane(layer, lane, fallbackLength = 1) {
    return getActivationData(layer, lane, 'getMlpUp', resolveBaseVectorLength(layer, fallbackLength) * 4);
}

export function getMlpActivationDataForLane(layer, lane, fallbackLength = 1) {
    return getActivationData(layer, lane, 'getMlpActivation', resolveBaseVectorLength(layer, fallbackLength) * 4);
}

export function getMlpDownDataForLane(layer, lane, fallbackLength = 1) {
    return getActivationData(layer, lane, 'getMlpDown', resolveBaseVectorLength(layer, fallbackLength));
}

export function getPostMlpResidualDataForLane(layer, lane, fallbackLength = 1) {
    return getActivationData(layer, lane, 'getPostMlpResidual', resolveBaseVectorLength(layer, fallbackLength));
}
