import { MHA_FINAL_Q_COLOR } from '../animations/LayerAnimationConstants.js';
import { getLayerNormParamMeta } from '../data/layerNormParams.js';
import { buildHueRangeOptions } from '../utils/colors.js';
import {
    isLayerNormNormalizedStage,
    resolveLayerNormKind
} from '../utils/layerNormLabels.js';
import {
    findUserDataNumber,
    findUserDataString,
    getActivationDataFromSelection
} from './selectionPanelSelectionUtils.js';
import { D_MODEL } from './selectionPanelConstants.js';
import { formatTokenLabelForPreview } from './selectionPanelFormatUtils.js';

export const LAYER_NORM_ACTIVE_PARAM_PREVIEW_COLOR_OPTIONS = Object.freeze(buildHueRangeOptions(
    MHA_FINAL_Q_COLOR,
    {
        hueSpread: 0.1,
        minLightness: 0.34,
        maxLightness: 0.74,
        valueMin: -1.8,
        valueMax: 1.8
    }
));

function resolveLayerFromEngine(engine = null, layerNormKind = null, layerIndex = null) {
    const layers = Array.isArray(engine?._layers) ? engine._layers : [];
    if (!layers.length) return null;

    if (layerNormKind === 'final') {
        return layers[layers.length - 1] || null;
    }

    if (!Number.isFinite(layerIndex)) return null;
    const safeIndex = Math.floor(layerIndex);
    if (safeIndex < 0 || safeIndex >= layers.length) return null;
    return layers[safeIndex] || null;
}

function getObjectInstancedLaneCount(object = null) {
    if (!object || typeof object.traverse !== 'function') return 0;
    let laneCount = 0;
    object.traverse((child) => {
        if (!child?.isInstancedMesh) return;
        const count = Number.isFinite(child.count)
            ? Math.floor(child.count)
            : Math.floor(child.instanceMatrix?.count || 0);
        if (count > laneCount) laneCount = count;
    });
    return laneCount;
}

function resolveLayerMethodValue(layer = null, methodName = '', fallbackKey = '') {
    if (layer && typeof layer[methodName] === 'function') {
        try {
            return layer[methodName]();
        } catch (_) {
            // Fall through to direct field access.
        }
    }
    return fallbackKey ? layer?.[fallbackKey] : undefined;
}

function resolveLayerLaneByToken(layer = null, selectionInfo = null) {
    const lanes = Array.isArray(layer?.lanes) ? layer.lanes : [];
    if (!lanes.length) return null;

    const tokenIndex = findUserDataNumber(selectionInfo, 'tokenIndex');
    if (Number.isFinite(tokenIndex)) {
        const normalizedTokenIndex = Math.floor(tokenIndex);
        const byIndex = lanes.find((lane) => (
            lane
            && Number.isFinite(lane.tokenIndex)
            && Math.floor(lane.tokenIndex) === normalizedTokenIndex
        ));
        if (byIndex) return byIndex;
    }

    const tokenLabelRaw = findUserDataString(selectionInfo, 'tokenLabel');
    const tokenLabel = formatTokenLabelForPreview(tokenLabelRaw);
    if (tokenLabel) {
        const byLabel = lanes.find((lane) => (
            lane
            && typeof lane.tokenLabel === 'string'
            && formatTokenLabelForPreview(lane.tokenLabel) === tokenLabel
        ));
        if (byLabel) return byLabel;
    }

    return null;
}

export function resolveLayerNormPreviewContext(selectionInfo = null, engine = null) {
    const label = selectionInfo?.label || '';
    const activationStage = String(getActivationDataFromSelection(selectionInfo)?.stage || '');
    const explicitKind = findUserDataString(selectionInfo, 'layerNormKind') || null;
    const inferredKind = resolveLayerNormKind({
        label,
        stage: activationStage,
        explicitKind
    });

    let layerIndex = findUserDataNumber(selectionInfo, 'layerIndex');
    const targetLayer = resolveLayerFromEngine(engine, inferredKind, layerIndex);
    if (!Number.isFinite(layerIndex) && Number.isFinite(targetLayer?.index)) {
        layerIndex = Math.floor(targetLayer.index);
    }

    const layoutValue = resolveLayerMethodValue(targetLayer, '_getLaneLayoutCount', '_laneLayoutCount');
    const activeLaneValues = resolveLayerMethodValue(
        targetLayer,
        '_getActiveLaneLayoutIndices',
        '_activeLaneLayoutIndices'
    );
    const baseVectorLengthValue = resolveLayerMethodValue(
        targetLayer,
        '_getBaseVectorLength',
        '_baseVectorLength'
    );
    const objectLaneCount = getObjectInstancedLaneCount(selectionInfo?.object || selectionInfo?.hit?.object);

    const layoutCount = Number.isFinite(layoutValue) && layoutValue > 0
        ? Math.floor(layoutValue)
        : (objectLaneCount > 0 ? objectLaneCount : 0);
    const safeLayoutCount = layoutCount > 0 ? layoutCount : 0;

    let activeLaneLayoutIndices = Array.isArray(activeLaneValues)
        ? activeLaneValues
            .map((value) => Number.isFinite(value) ? Math.floor(value) : null)
            .filter((value) => value !== null)
        : [];
    if (!activeLaneLayoutIndices.length && safeLayoutCount > 0) {
        activeLaneLayoutIndices = Array.from({ length: safeLayoutCount }, (_, idx) => idx);
    }
    if (safeLayoutCount > 0) {
        activeLaneLayoutIndices = activeLaneLayoutIndices.map((value) => (
            Math.max(0, Math.min(safeLayoutCount - 1, value))
        ));
    }

    const baseVectorLength = Number.isFinite(baseVectorLengthValue) && baseVectorLengthValue > 0
        ? Math.floor(baseVectorLengthValue)
        : D_MODEL;

    return {
        layerNormKind: inferredKind,
        layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
        layoutCount: safeLayoutCount,
        activeLaneLayoutIndices,
        baseVectorLength,
        targetLayer
    };
}

export function resolveLayerNormParameterSummary(selectionInfo = null, engine = null) {
    const previewContext = resolveLayerNormPreviewContext(selectionInfo, engine);
    const layerNormParamMeta = getLayerNormParamMeta();
    const hiddenSize = Number.isFinite(layerNormParamMeta?.hidden_size) && layerNormParamMeta.hidden_size > 0
        ? Math.floor(layerNormParamMeta.hidden_size)
        : D_MODEL;

    return {
        ...previewContext,
        perParameterCount: hiddenSize,
        totalParameterCount: hiddenSize * 2
    };
}

export function resolveLiveLayerNormNormalizedPreviewSelection(selectionInfo = null, engine = null) {
    const activationStage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    if (!isLayerNormNormalizedStage(activationStage)) return null;

    const previewContext = resolveLayerNormPreviewContext(selectionInfo, engine);
    if (!previewContext?.targetLayer || previewContext.layerNormKind === 'final') return null;

    const lane = resolveLayerLaneByToken(previewContext.targetLayer, selectionInfo);
    if (!lane) return null;

    let vectorRef = null;
    let stageIsActive = false;
    if (activationStage === 'ln1.norm') {
        vectorRef = lane.dupVec || null;
        stageIsActive = !!(lane.normStarted && !lane.multStarted);
    } else if (activationStage === 'ln2.norm') {
        vectorRef = lane.movingVecLN2 || null;
        stageIsActive = !!(lane.normStartedLN2 && !lane.multDoneLN2);
    }
    if (!stageIsActive || !vectorRef || vectorRef.group?.visible === false) return null;

    const object = vectorRef.mesh?.isInstancedMesh === true
        ? vectorRef.mesh
        : (vectorRef.group || vectorRef.mesh || null);
    if (!object) return null;

    const info = {
        ...((selectionInfo?.info && typeof selectionInfo.info === 'object') ? selectionInfo.info : {}),
        vectorRef
    };
    if (Number.isFinite(vectorRef?.instanceCount)) {
        info.prismCount = Math.max(1, Math.floor(vectorRef.instanceCount));
        info.prismIndex = 0;
    }

    return {
        ...(selectionInfo && typeof selectionInfo === 'object' ? selectionInfo : {}),
        info,
        object,
        ...(vectorRef.mesh?.isInstancedMesh === true ? {
            hit: {
                object,
                instanceId: 0
            }
        } : {})
    };
}
