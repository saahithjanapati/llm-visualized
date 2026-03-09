import { PRISM_DIMENSIONS_PER_UNIT } from '../utils/constants.js';
import { D_HEAD, D_MODEL } from './selectionPanelConstants.js';
import { getActivationDataFromSelection } from './selectionPanelSelectionUtils.js';

const DEFAULT_VECTOR_STRIDE = Math.max(1, Math.floor(PRISM_DIMENSIONS_PER_UNIT || 64));
const MLP_SEGMENT_COUNT = 4;

function clampPositiveInt(value, fallback) {
    const next = Number(value);
    if (!Number.isFinite(next) || next <= 0) return fallback;
    return Math.max(1, Math.floor(next));
}

function toNumericArray(values) {
    if (!Array.isArray(values) && !ArrayBuffer.isView(values)) return null;
    if (values.length <= 0) return null;
    return Array.from(values, (value) => {
        const num = Number(value);
        return Number.isFinite(num) ? num : 0;
    });
}

function resolveConfiguredStride(config, key) {
    return clampPositiveInt(config?.[key], DEFAULT_VECTOR_STRIDE);
}

function resolveSamplingFamily(label = '', stage = '') {
    const lower = String(label || '').toLowerCase();
    const stageLower = String(stage || '').toLowerCase();

    if (
        stageLower.startsWith('mlp.up')
        || stageLower.startsWith('mlp.activation')
        || lower.includes('mlp up projection')
        || lower.includes('mlp activation')
        || lower.includes('mlp expanded segments')
    ) {
        return 'mlp';
    }

    if (
        stageLower.startsWith('qkv.')
        || stageLower === 'attention.weighted_value'
        || lower.includes('attention weighted sum')
        || lower.includes('weighted value vector')
        || lower.includes('query vector')
        || lower.includes('key vector')
        || lower.includes('value vector')
        || lower.includes('merged key vectors')
        || lower.includes('merged value vectors')
    ) {
        return 'attention';
    }

    if (
        stageLower.startsWith('embedding.')
        || stageLower.startsWith('layer.incoming')
        || stageLower.startsWith('ln1.')
        || stageLower.startsWith('ln2.')
        || stageLower.startsWith('final_ln.')
        || stageLower.startsWith('attention.output_projection')
        || stageLower.startsWith('residual.post_attention')
        || stageLower.startsWith('mlp.down')
        || stageLower.startsWith('residual.post_mlp')
        || stageLower.includes('.param.')
        || lower.includes('residual')
        || lower.includes('layernorm scale')
        || lower.includes('layernorm shift')
        || lower.includes('layer norm scale')
        || lower.includes('layer norm shift')
    ) {
        return 'residual';
    }

    return null;
}

function resolveVectorDomain(family, segmentIndex = null) {
    if (family === 'attention') {
        return {
            originalLength: D_HEAD,
            domainLength: D_HEAD,
            offset: 0,
            title: 'Attention head sampling'
        };
    }

    if (family === 'mlp') {
        const safeSegmentIndex = Number.isFinite(segmentIndex)
            ? Math.max(0, Math.min(MLP_SEGMENT_COUNT - 1, Math.floor(segmentIndex)))
            : null;
        if (safeSegmentIndex !== null) {
            return {
                originalLength: D_MODEL * MLP_SEGMENT_COUNT,
                domainLength: D_MODEL,
                offset: safeSegmentIndex * D_MODEL,
                title: 'MLP segment sampling',
                segmentIndex: safeSegmentIndex
            };
        }
        return {
            originalLength: D_MODEL * MLP_SEGMENT_COUNT,
            domainLength: D_MODEL * MLP_SEGMENT_COUNT,
            offset: 0,
            title: 'MLP expansion sampling',
            segmentIndex: null
        };
    }

    return {
        originalLength: D_MODEL,
        domainLength: D_MODEL,
        offset: 0,
        title: 'Vector sampling'
    };
}

function resolveEffectiveStride(domainLength, valueCount, configuredStride) {
    const safeDomainLength = clampPositiveInt(domainLength, D_MODEL);
    const safeValueCount = clampPositiveInt(valueCount, 1);
    const safeConfiguredStride = clampPositiveInt(configuredStride, DEFAULT_VECTOR_STRIDE);
    const configuredCount = Math.ceil(safeDomainLength / safeConfiguredStride);

    if (safeValueCount >= safeDomainLength) return 1;
    if (safeValueCount === configuredCount) return safeConfiguredStride;

    return Math.max(1, Math.ceil(safeDomainLength / safeValueCount));
}

function formatNumericValue(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '0.0000';
    return num.toFixed(4);
}

export function resolveSelectionVectorSamplingData({
    label = '',
    selectionInfo = null,
    activationSource = null,
    fallbackValues = null
} = {}) {
    const activationData = getActivationDataFromSelection(selectionInfo);
    const values = toNumericArray(activationData?.values) || toNumericArray(fallbackValues);
    if (!values) return null;

    const family = resolveSamplingFamily(label, activationData?.stage);
    if (!family) return null;

    const config = activationSource?.meta?.config || null;
    const configuredStride = family === 'attention'
        ? resolveConfiguredStride(config, 'attention_stride')
        : family === 'mlp'
            ? resolveConfiguredStride(config, 'mlp_stride')
            : resolveConfiguredStride(config, 'residual_stride');
    const domain = resolveVectorDomain(family, activationData?.segmentIndex);
    const effectiveStride = resolveEffectiveStride(domain.domainLength, values.length, configuredStride);
    const maxSourceIndex = domain.offset + domain.domainLength - 1;
    const entries = values.map((value, index) => {
        const localSourceIndex = Math.min(domain.domainLength - 1, index * effectiveStride);
        return {
            sourceIndex: Math.min(maxSourceIndex, domain.offset + localSourceIndex),
            value
        };
    });

    const lines = [
        `Original vector length: ${domain.originalLength.toLocaleString('en-US')}`,
        `Samples used for color: ${values.length.toLocaleString('en-US')}`,
        `Sampling stride: ${effectiveStride.toLocaleString('en-US')}`
    ];

    if (Number.isFinite(domain.segmentIndex)) {
        lines.splice(
            1,
            0,
            `MLP segment: ${domain.segmentIndex + 1}/${MLP_SEGMENT_COUNT} (dims ${domain.offset}-${maxSourceIndex})`
        );
    }

    lines.push('');
    entries.forEach((entry) => {
        lines.push(`dim ${entry.sourceIndex}: ${formatNumericValue(entry.value)}`);
    });

    return {
        title: domain.title,
        text: lines.join('\n')
    };
}
