import { PRISM_DIMENSIONS_PER_UNIT } from '../utils/constants.js';
import { D_HEAD, D_MODEL } from './selectionPanelConstants.js';
import { getActivationDataFromSelection } from './selectionPanelSelectionUtils.js';

const DEFAULT_VECTOR_STRIDE = Math.max(1, Math.floor(PRISM_DIMENSIONS_PER_UNIT || 64));
const MLP_SEGMENT_COUNT = 4;
const VECTOR_SAMPLING_VALUE_DECIMALS = 3;

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

function extractMlpExpandedSamplingValues(selectionInfo = null) {
    const vectorRef = selectionInfo?.info?.vectorRef || null;
    const batchRefs = vectorRef?.isBatchedVectorRef && Array.isArray(vectorRef?._batch?._vectorRefs)
        ? vectorRef._batch._vectorRefs
        : null;
    if (!batchRefs?.length) return null;

    const combined = [];
    const orderedRefs = batchRefs
        .filter((candidate) => candidate && typeof candidate === 'object')
        .slice()
        .sort((a, b) => {
            const left = Number.isFinite(a?._index) ? Math.floor(a._index) : 0;
            const right = Number.isFinite(b?._index) ? Math.floor(b._index) : 0;
            return left - right;
        });

    orderedRefs.forEach((candidate) => {
        const segmentValues = toNumericArray(candidate?.userData?.activationData?.values)
            || toNumericArray(candidate?.rawData)
            || toNumericArray(candidate?.group?.userData?.activationData?.values)
            || toNumericArray(candidate?.mesh?.userData?.activationData?.values);
        if (segmentValues?.length) {
            combined.push(...segmentValues);
        }
    });

    return combined.length ? combined : null;
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
        || stageLower === 'attention.weighted_sum'
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

function isWeightedSumSampling(label = '', stage = '') {
    const lower = String(label || '').toLowerCase();
    const stageLower = String(stage || '').toLowerCase();
    return stageLower === 'attention.weighted_sum' || lower.includes('attention weighted sum');
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
    if (!Number.isFinite(num)) return '0';
    const rounded = Number(num.toFixed(VECTOR_SAMPLING_VALUE_DECIMALS));
    const normalized = Object.is(rounded, -0) ? 0 : rounded;
    return normalized
        .toFixed(VECTOR_SAMPLING_VALUE_DECIMALS)
        .replace(/\.?0+$/, '');
}

function buildVectorSamplingText(summaryRows = [], tableRows = [], description = '') {
    const lines = [];
    if (typeof description === 'string' && description.trim().length) {
        lines.push(description.trim(), '');
    }
    lines.push(...summaryRows.map((row) => `${row.label}: ${row.value}`));
    if (tableRows.length) lines.push('');
    tableRows.forEach((row) => {
        lines.push(`dim ${row.dimension}: ${row.value}`);
    });
    return lines.join('\n');
}

function buildSamplingDescription(family, domain) {
    const roundedNote = 'Values are rounded for readability.';
    if (family === 'attention') {
        return `These rows show the sampled source dimensions for this attention head vector. They drive the colors in the scene, not the full dense tensor. ${roundedNote}`;
    }
    if (family === 'mlp' && Number.isFinite(domain?.segmentIndex)) {
        return `These rows show the sampled source dimensions for this MLP segment within the 3,072-d expansion. They are the debug subset used for the current colors, not the full tensor. ${roundedNote}`;
    }
    if (family === 'mlp') {
        return `These rows show the sampled source dimensions for the expanded MLP vector. They are the debug subset used for the current colors, not the full tensor. ${roundedNote}`;
    }
    return `These rows show the sampled source dimensions for the model-width vector behind this selection. They drive the colors in the scene, not the full dense tensor. ${roundedNote}`;
}

export function renderSelectionVectorSamplingData(container, data = null) {
    if (!container || typeof document === 'undefined') return;
    container.replaceChildren();
    if (!data) return;

    const fragment = document.createDocumentFragment();

    if (Array.isArray(data.summaryRows) && data.summaryRows.length) {
        const summaryEl = document.createElement('div');
        summaryEl.className = 'detail-data-summary';
        data.summaryRows.forEach((row) => {
            const rowEl = document.createElement('div');
            rowEl.className = 'detail-data-summary-row';

            const labelEl = document.createElement('span');
            labelEl.className = 'detail-data-summary-label';
            labelEl.textContent = row.label;

            const valueEl = document.createElement('span');
            valueEl.className = 'detail-data-summary-value';
            valueEl.textContent = row.value;

            rowEl.append(labelEl, valueEl);
            summaryEl.appendChild(rowEl);
        });
        fragment.appendChild(summaryEl);
    }

    if (Array.isArray(data.tableRows) && data.tableRows.length) {
        const tableWrap = document.createElement('div');
        tableWrap.className = 'detail-data-table-wrap';

        const table = document.createElement('table');
        table.className = 'detail-data-table';

        const thead = document.createElement('thead');
        const headRow = document.createElement('tr');
        ['Dimension', 'Value'].forEach((label) => {
            const th = document.createElement('th');
            th.scope = 'col';
            th.textContent = label;
            headRow.appendChild(th);
        });
        thead.appendChild(headRow);

        const tbody = document.createElement('tbody');
        data.tableRows.forEach((row) => {
            const tr = document.createElement('tr');

            const dimensionEl = document.createElement('td');
            dimensionEl.className = 'detail-data-table-dimension';
            dimensionEl.textContent = row.dimension;

            const valueEl = document.createElement('td');
            valueEl.className = 'detail-data-table-value';
            valueEl.textContent = row.value;

            tr.append(dimensionEl, valueEl);
            tbody.appendChild(tr);
        });

        table.append(thead, tbody);
        tableWrap.appendChild(table);
        fragment.appendChild(tableWrap);
    }

    container.appendChild(fragment);
}

export function resolveSelectionVectorSamplingData({
    label = '',
    selectionInfo = null,
    activationSource = null,
    fallbackValues = null
} = {}) {
    const activationData = getActivationDataFromSelection(selectionInfo);
    const activationValues = toNumericArray(activationData?.values);
    const fallbackNumericValues = toNumericArray(fallbackValues);
    const family = resolveSamplingFamily(label, activationData?.stage);
    if (!family) return null;
    const mlpExpandedValues = family === 'mlp'
        ? extractMlpExpandedSamplingValues(selectionInfo)
        : null;
    const values = mlpExpandedValues || activationValues || fallbackNumericValues;
    if (!values) return null;

    const config = activationSource?.meta?.config || null;
    const configuredStride = family === 'attention'
        ? resolveConfiguredStride(config, 'attention_stride')
        : family === 'mlp'
            ? resolveConfiguredStride(config, 'mlp_stride')
            : resolveConfiguredStride(config, 'residual_stride');
    const segmentIndex = family === 'mlp' && mlpExpandedValues
        ? null
        : activationData?.segmentIndex;
    const domain = resolveVectorDomain(family, segmentIndex);
    const weightedSumSampling = family === 'attention'
        && isWeightedSumSampling(label, activationData?.stage)
        && !!activationValues;
    const effectiveStride = resolveEffectiveStride(domain.domainLength, values.length, configuredStride);
    const displayedValues = weightedSumSampling
        ? values.slice(0, 1)
        : values;
    const displayedStride = weightedSumSampling
        ? Math.max(1, domain.domainLength)
        : effectiveStride;
    const maxSourceIndex = domain.offset + domain.domainLength - 1;
    const description = weightedSumSampling
        ? `This weighted sum currently renders as one grouped prism, so only the first sampled head dimension is used for its dev-mode color readout. Values are rounded for readability.`
        : buildSamplingDescription(family, domain);
    const entries = displayedValues.map((value, index) => {
        const localSourceIndex = Math.min(domain.domainLength - 1, index * displayedStride);
        return {
            sourceIndex: Math.min(maxSourceIndex, domain.offset + localSourceIndex),
            value
        };
    });

    const summaryRows = [
        { label: 'Original vector length', value: domain.originalLength.toLocaleString('en-US') },
        { label: 'Samples used for color', value: displayedValues.length.toLocaleString('en-US') },
        { label: 'Sampling stride', value: displayedStride.toLocaleString('en-US') }
    ];

    if (Number.isFinite(domain.segmentIndex)) {
        summaryRows.splice(
            1,
            0,
            {
                label: 'MLP segment',
                value: `${domain.segmentIndex + 1}/${MLP_SEGMENT_COUNT} (dims ${domain.offset.toLocaleString('en-US')}-${maxSourceIndex.toLocaleString('en-US')})`
            }
        );
    }

    const tableRows = entries.map((entry) => ({
        sourceIndex: entry.sourceIndex,
        dimension: entry.sourceIndex.toLocaleString('en-US'),
        value: formatNumericValue(entry.value)
    }));

    return {
        title: domain.title,
        description,
        summaryRows,
        tableRows,
        text: buildVectorSamplingText(summaryRows, tableRows, description)
    };
}
