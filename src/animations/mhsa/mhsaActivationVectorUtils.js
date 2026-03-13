import { NUM_HEAD_SETS_LAYER } from '../../utils/constants.js';
import { buildActivationData, applyActivationDataToVector } from '../../utils/activationMetadata.js';

function toNumericArray(values) {
    if (!Array.isArray(values) && !ArrayBuffer.isView(values)) return null;
    if (values.length === 0) return null;
    return Array.from(values, (value) => (Number.isFinite(value) ? value : 0));
}

export function applyAttentionOutputProjectionDataToVector(
    vectorRef,
    {
        activationSource = null,
        layerIndex = null,
        tokenIndex = null,
        tokenLabel = null,
        vectorPrismCount = null,
        outputVectorLength = 64,
        fallbackData = null
    } = {}
) {
    if (!vectorRef || typeof vectorRef.applyProcessedVisuals !== 'function') return null;
    const outputData = activationSource
        && Number.isFinite(layerIndex)
        && Number.isFinite(tokenIndex)
        && typeof activationSource.getAttentionOutputProjection === 'function'
        ? activationSource.getAttentionOutputProjection(layerIndex, tokenIndex, vectorPrismCount)
        : null;
    const newRaw = toNumericArray(outputData) || toNumericArray(fallbackData);
    if (!newRaw) return null;

    const numKeyColors = Math.min(30, Math.max(1, newRaw.length || 1));
    const numVisibleOutputUnits = Math.max(1, Math.floor(outputVectorLength || 64)) * NUM_HEAD_SETS_LAYER;
    vectorRef.applyProcessedVisuals(
        newRaw,
        numVisibleOutputUnits,
        { numKeyColors, generationOptions: null },
        { setHiddenToBlack: false },
        outputData || null
    );

    const label = tokenLabel
        ? `Attention Output Projection - ${tokenLabel}`
        : 'Attention Output Projection';
    const activationData = buildActivationData({
        label,
        stage: 'attention.output_projection',
        layerIndex,
        tokenIndex,
        tokenLabel,
        values: newRaw,
        copyValues: false,
    });
    applyActivationDataToVector(vectorRef, activationData, label);
    return newRaw;
}

export function applyPostAttentionResidualDataToVector(
    vectorRef,
    {
        values = null,
        layerIndex = null,
        tokenIndex = null,
        tokenLabel = null
    } = {}
) {
    const raw = toNumericArray(values);
    if (!vectorRef || !raw) return false;

    vectorRef.rawData = raw.slice();
    if (typeof vectorRef.updateKeyColorsFromData === 'function') {
        const numKeyColors = Math.min(30, Math.max(1, raw.length || 1));
        vectorRef.updateKeyColorsFromData(vectorRef.rawData, numKeyColors, null, raw);
    }

    const label = tokenLabel
        ? `Post-Attention Residual - ${tokenLabel}`
        : 'Post-Attention Residual';
    const activationData = buildActivationData({
        label,
        stage: 'residual.post_attention',
        layerIndex,
        tokenIndex,
        tokenLabel,
        values: raw,
        copyValues: false,
    });
    applyActivationDataToVector(vectorRef, activationData, label);
    return true;
}
