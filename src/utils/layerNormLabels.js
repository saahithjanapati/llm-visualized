function normalizeLayerNormKind(kind = null) {
    const lower = String(kind || '').toLowerCase();
    if (lower === 'ln1' || lower === 'layernorm1' || lower === 'layer_norm_1') return 'ln1';
    if (lower === 'ln2' || lower === 'layernorm2' || lower === 'layer_norm_2') return 'ln2';
    if (lower === 'final' || lower === 'top') return 'final';
    return null;
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

export function formatLayerNormLabel(kind = null) {
    const safeKind = normalizeLayerNormKind(kind);
    if (safeKind === 'ln1') return 'LayerNorm 1';
    if (safeKind === 'ln2') return 'LayerNorm 2';
    if (safeKind === 'final') return 'LayerNorm (Top)';
    return 'LayerNorm';
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
