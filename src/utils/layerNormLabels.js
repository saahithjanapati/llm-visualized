function normalizeLayerNormKind(kind = null) {
    const lower = String(kind || '').toLowerCase();
    if (lower === 'ln1' || lower === 'layernorm1' || lower === 'layer_norm_1') return 'ln1';
    if (lower === 'ln2' || lower === 'layernorm2' || lower === 'layer_norm_2') return 'ln2';
    if (lower === 'final' || lower === 'top') return 'final';
    return null;
}

function normalizePostLayerNormResidualKind(kind = null) {
    const safeKind = normalizeLayerNormKind(kind);
    return safeKind === 'ln1' || safeKind === 'ln2' ? safeKind : null;
}

const POST_LAYER_NORM_OUTPUT_STAGE_KIND = Object.freeze({
    'ln1.output': 'ln1',
    'ln1.shift': 'ln1',
    'ln2.output': 'ln2',
    'ln2.shift': 'ln2'
});

const LAYER_NORM_OUTPUT_STAGE_KIND = Object.freeze({
    ...POST_LAYER_NORM_OUTPUT_STAGE_KIND,
    'final_ln.output': 'final',
    'final_ln.shift': 'final'
});

const LAYER_NORM_NORMALIZED_STAGE_KIND = Object.freeze({
    'ln1.norm': 'ln1',
    'ln2.norm': 'ln2',
    'final_ln.norm': 'final'
});

const LAYER_NORM_PRODUCT_STAGE_KIND = Object.freeze({
    'ln1.scale': 'ln1',
    'ln1.product': 'ln1',
    'ln2.scale': 'ln2',
    'ln2.product': 'ln2',
    'final_ln.scale': 'final',
    'final_ln.product': 'final'
});

function matchPostLayerNormResidualLabel(label = '') {
    return String(label || '')
        .toLowerCase()
        .match(/\bpost[-\s]*layer\s*norm(?:\s*([12]))?\s+residual\b/);
}

function normalizeLayerNormParam(param = null) {
    const lower = String(param || '').toLowerCase();
    if (lower === 'scale' || lower === 'gamma') return 'scale';
    if (lower === 'shift' || lower === 'beta') return 'shift';
    return null;
}

export function normalizePostLayerNormResidualStage(stage = '', {
    preferLegacy = false
} = {}) {
    const stageLower = String(stage || '').toLowerCase();
    const kind = POST_LAYER_NORM_OUTPUT_STAGE_KIND[stageLower] || null;
    if (!kind) return '';
    return preferLegacy
        ? `${kind}.shift`
        : `${kind}.output`;
}

export function isPostLayerNormResidualStage(stage = '') {
    return !!normalizePostLayerNormResidualStage(stage);
}

export function normalizeLayerNormOutputStage(stage = '', {
    preferLegacy = false
} = {}) {
    const stageLower = String(stage || '').toLowerCase();
    const kind = LAYER_NORM_OUTPUT_STAGE_KIND[stageLower] || null;
    if (!kind) return '';
    if (preferLegacy) {
        return kind === 'final'
            ? 'final_ln.shift'
            : `${kind}.shift`;
    }
    return kind === 'final'
        ? 'final_ln.output'
        : `${kind}.output`;
}

export function isLayerNormOutputStage(stage = '') {
    return !!normalizeLayerNormOutputStage(stage);
}

export function isLayerNormNormalizedStage(stage = '') {
    return !!LAYER_NORM_NORMALIZED_STAGE_KIND[String(stage || '').toLowerCase()];
}

export function normalizeLayerNormProductStage(stage = '', {
    preferLegacy = false
} = {}) {
    const stageLower = String(stage || '').toLowerCase();
    const kind = LAYER_NORM_PRODUCT_STAGE_KIND[stageLower] || null;
    if (!kind) return '';
    if (preferLegacy) {
        return kind === 'final'
            ? 'final_ln.scale'
            : `${kind}.scale`;
    }
    return kind === 'final'
        ? 'final_ln.product'
        : `${kind}.product`;
}

export function isLayerNormProductStage(stage = '') {
    return !!normalizeLayerNormProductStage(stage);
}

export function resolveLayerNormKind({
    label = '',
    stage = '',
    explicitKind = null
} = {}) {
    const safeExplicit = normalizeLayerNormKind(explicitKind);
    if (safeExplicit) return safeExplicit;

    const stageLower = String(stage || '').toLowerCase();
    if (stageLower.startsWith('ln1.')) return 'ln1';
    if (stageLower.startsWith('ln2.')) return 'ln2';
    if (stageLower.startsWith('final_ln')) return 'final';

    const lower = String(label || '').toLowerCase();
    if (
        lower.includes('layernorm (top)')
        || lower.includes('top layernorm')
        || lower.includes('final ln')
        || lower.includes('final layernorm')
    ) {
        return 'final';
    }
    if (
        /\bln1\b/.test(lower)
        || lower.includes('layernorm 1')
        || lower.includes('layer norm 1')
        || lower.includes('first layernorm')
        || lower.includes('first layer norm')
    ) {
        return 'ln1';
    }
    if (
        /\bln2\b/.test(lower)
        || lower.includes('layernorm 2')
        || lower.includes('layer norm 2')
        || lower.includes('second layernorm')
        || lower.includes('second layer norm')
    ) {
        return 'ln2';
    }
    return null;
}

export function isPostLayerNormResidualSelection({
    label = '',
    stage = ''
} = {}) {
    return isPostLayerNormResidualStage(stage)
        || Boolean(matchPostLayerNormResidualLabel(label));
}

export function resolvePostLayerNormResidualKind({
    label = '',
    stage = '',
    explicitKind = null
} = {}) {
    const safeExplicit = normalizePostLayerNormResidualKind(explicitKind);
    const normalizedStage = normalizePostLayerNormResidualStage(stage);
    if (normalizedStage === 'ln1.output') return 'ln1';
    if (normalizedStage === 'ln2.output') return 'ln2';

    const labelMatch = matchPostLayerNormResidualLabel(label);
    if (!labelMatch) return null;
    if (labelMatch[1] === '1') return 'ln1';
    if (labelMatch[1] === '2') return 'ln2';
    if (safeExplicit) return safeExplicit;

    return normalizePostLayerNormResidualKind(resolveLayerNormKind({
        label,
        stage,
        explicitKind
    }));
}

export function formatPostLayerNormResidualLabel(kind = null) {
    const safeKind = normalizePostLayerNormResidualKind(kind);
    if (safeKind === 'ln1') return 'Post LayerNorm 1 Residual Vector';
    if (safeKind === 'ln2') return 'Post LayerNorm 2 Residual Vector';
    return 'Post LayerNorm Residual Vector';
}

export function resolvePostLayerNormResidualLabel({
    label = '',
    stage = '',
    explicitKind = null
} = {}) {
    return formatPostLayerNormResidualLabel(resolvePostLayerNormResidualKind({
        label,
        stage,
        explicitKind
    }));
}

export function formatLayerNormLabel(kind = null) {
    const safeKind = normalizeLayerNormKind(kind);
    if (safeKind === 'ln1') return 'LayerNorm 1';
    if (safeKind === 'ln2') return 'LayerNorm 2';
    if (safeKind === 'final') return 'LayerNorm (Top)';
    return 'LayerNorm';
}

export function formatNormalizedResidualStreamLabel() {
    return 'Normalized Residual Stream Vector';
}

export function formatLayerNormParamLabel(kind = null, param = 'scale') {
    const safeParam = String(param || '').toLowerCase() === 'shift' ? 'Shift' : 'Scale';
    const safeKind = normalizeLayerNormKind(kind);
    if (safeKind === 'final') {
        return `Final LN ${safeParam}`;
    }
    if (safeKind === 'ln1' || safeKind === 'ln2') {
        return `${formatLayerNormLabel(safeKind)} ${safeParam}`;
    }
    return `LayerNorm ${safeParam}`;
}

export function formatLayerNormProductVectorLabel(kind = null) {
    const safeKind = normalizeLayerNormKind(kind);
    if (safeKind === 'ln1' || safeKind === 'ln2' || safeKind === 'final') {
        return `${formatLayerNormLabel(safeKind)} Product Vector`;
    }
    return 'LayerNorm Product Vector';
}

export function resolveLayerNormProductVectorLabel({
    label = '',
    stage = '',
    explicitKind = null
} = {}) {
    return formatLayerNormProductVectorLabel(resolveLayerNormKind({
        label,
        stage,
        explicitKind
    }));
}

export function resolveLayerNormParamSpec({
    label = '',
    stage = '',
    explicitKind = null,
    explicitParam = null
} = {}) {
    const layerNormKind = resolveLayerNormKind({ label, stage, explicitKind });
    if (!layerNormKind) return null;

    const safeParam = normalizeLayerNormParam(explicitParam);
    if (safeParam) {
        return { layerNormKind, param: safeParam };
    }

    const lower = String(label || '').toLowerCase();
    const stageLower = String(stage || '').toLowerCase();
    const hasScaleLabel = /\blayer\s*norm(?:\s*[12]|\s*\(top\))?\s+scale\b/.test(lower)
        || /\bfinal\s+ln\s+scale\b/.test(lower)
        || /\bln[12]\s+scale\b/.test(lower);
    const hasShiftLabel = /\blayer\s*norm(?:\s*[12]|\s*\(top\))?\s+shift\b/.test(lower)
        || /\bfinal\s+ln\s+shift\b/.test(lower)
        || /\bln[12]\s+shift\b/.test(lower);
    const isScale = stageLower.endsWith('.param.scale')
        || hasScaleLabel;
    const isShift = stageLower.endsWith('.param.shift')
        || hasShiftLabel;
    const param = isScale ? 'scale' : (isShift ? 'shift' : null);

    return param ? { layerNormKind, param } : null;
}

export function expandLayerNormLabel(label = '', kind = null) {
    const raw = String(label || '');
    if (!raw.trim().length) return raw;

    const safeKind = normalizeLayerNormKind(kind) || resolveLayerNormKind({ label: raw });
    if (/^layer\s*norm(?:\s*\(top\))?$/i.test(raw)) {
        return formatLayerNormLabel(safeKind);
    }

    let expanded = raw
        .replace(/\bLN1\b/g, 'LayerNorm 1')
        .replace(/\bLN2\b/g, 'LayerNorm 2');

    if (safeKind === 'ln1' || safeKind === 'ln2') {
        const baseLabel = formatLayerNormLabel(safeKind);
        if (
            /^layer\s*norm\b/i.test(expanded)
            && !/^layer\s*norm\s*[12]\b/i.test(expanded)
            && !/\(top\)/i.test(expanded)
        ) {
            expanded = expanded.replace(/^layer\s*norm\b/i, baseLabel);
        }
    }

    return expanded;
}
