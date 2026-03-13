const MHSA_TOKEN_MATRIX_PROJECTION_STAGE_SELECTOR = '[data-mhsa-projection-stage-index]';

const MHSA_TOKEN_MATRIX_PROJECTION_STAGE_COMPONENT_SPECS = Object.freeze([
    {
        key: 'input',
        selector: '.mhsa-token-matrix-preview__x-block'
    },
    {
        key: 'weight',
        selector: '.mhsa-token-matrix-preview__weight-block'
    },
    {
        key: 'bias',
        selector: '.mhsa-token-matrix-preview__bias-block'
    },
    {
        key: 'output',
        selector: '.mhsa-token-matrix-preview__query-block'
    },
    {
        key: 'multiply',
        selector: '.mhsa-token-matrix-preview__operator--matrix, .mhsa-token-matrix-preview__operator--xw'
    },
    {
        key: 'plus',
        selector: '.mhsa-token-matrix-preview__operator--wb'
    },
    {
        key: 'equals',
        selector: '.mhsa-token-matrix-preview__operator--bq'
    }
]);

const MHSA_TOKEN_MATRIX_PINNABLE_PROJECTION_STAGE_COMPONENT_SELECTOR = MHSA_TOKEN_MATRIX_PROJECTION_STAGE_COMPONENT_SPECS
    .map((spec) => spec.selector)
    .join(', ');

const MHSA_FULL_PROJECTION_STAGE_COMPONENT_KEYS = Object.freeze([
    'input',
    'multiply',
    'weight',
    'plus',
    'bias',
    'equals',
    'output'
]);

export function shouldMirrorMhsaHeadOutputRowFocus({
    hoverSource = '',
    projectionKind = ''
} = {}) {
    const safeHoverSource = String(hoverSource || '').toLowerCase();
    return (safeHoverSource === 'query' || safeHoverSource === 'x')
        && String(projectionKind || '').toLowerCase() === 'q';
}

export function buildMhsaProjectionComponentStateKey(stageIndex = null, componentKey = '') {
    if (!Number.isFinite(stageIndex)) return '';
    const safeComponentKey = String(componentKey || '').trim().toLowerCase();
    if (!safeComponentKey.length) return '';
    return `${Math.max(0, Math.floor(stageIndex))}:${safeComponentKey}`;
}

export function resolveMhsaProjectionWeightSceneFocus({
    projectionKind = '',
    stageIndex = null,
    firstProjectionStageIndex = null
} = {}) {
    const safeProjectionKind = String(projectionKind || '').trim().toLowerCase();
    if (!Number.isFinite(stageIndex)) return null;
    if (safeProjectionKind !== 'q' && safeProjectionKind !== 'k' && safeProjectionKind !== 'v') {
        return null;
    }

    const safeStageIndex = Math.max(0, Math.floor(stageIndex));
    const safeFirstProjectionStageIndex = Number.isFinite(firstProjectionStageIndex)
        ? Math.max(0, Math.floor(firstProjectionStageIndex))
        : null;
    const projectionComponents = [];
    const seen = new Set();
    const pushComponent = (nextStageIndex, componentKey) => {
        const stateKey = buildMhsaProjectionComponentStateKey(nextStageIndex, componentKey);
        if (!stateKey || seen.has(stateKey)) return;
        seen.add(stateKey);
        projectionComponents.push(stateKey);
    };

    if (Number.isFinite(safeFirstProjectionStageIndex)) {
        pushComponent(safeFirstProjectionStageIndex, 'input');
    }

    ['input', 'weight', 'output'].forEach((componentKey) => {
        pushComponent(safeStageIndex, componentKey);
    });
    const attentionBlocks = safeProjectionKind === 'q'
        ? ['query']
        : (safeProjectionKind === 'k'
            ? ['transpose']
            : ['value-post']);

    return {
        projectionComponents,
        attentionBlocks,
        connectors: [safeProjectionKind]
    };
}

export function resolveMhsaProjectionInputRowSceneFocus({
    stageIndex = null,
    firstProjectionStageIndex = null
} = {}) {
    if (!Number.isFinite(stageIndex)) return null;

    const safeStageIndex = Math.max(0, Math.floor(stageIndex));
    const safeFirstProjectionStageIndex = Number.isFinite(firstProjectionStageIndex)
        ? Math.max(0, Math.floor(firstProjectionStageIndex))
        : null;
    const projectionComponents = [];
    const seen = new Set();
    const pushComponent = (nextStageIndex, componentKey) => {
        const stateKey = buildMhsaProjectionComponentStateKey(nextStageIndex, componentKey);
        if (!stateKey || seen.has(stateKey)) return;
        seen.add(stateKey);
        projectionComponents.push(stateKey);
    };

    if (Number.isFinite(safeFirstProjectionStageIndex)) {
        pushComponent(safeFirstProjectionStageIndex, 'input');
    }

    MHSA_FULL_PROJECTION_STAGE_COMPONENT_KEYS.forEach((componentKey) => {
        pushComponent(safeStageIndex, componentKey);
    });

    return {
        projectionComponents
    };
}

export function resolveMhsaKeyRowSceneFocus({
    keyStageIndex = null,
    firstProjectionStageIndex = null
} = {}) {
    const projectionFocus = resolveMhsaProjectionInputRowSceneFocus({
        stageIndex: keyStageIndex,
        firstProjectionStageIndex
    });
    if (!projectionFocus) return null;
    return {
        ...projectionFocus,
        attentionBlocks: ['transpose'],
        connectors: ['k']
    };
}

export function resolveMhsaValueRowSceneFocus({
    valueStageIndex = null,
    firstProjectionStageIndex = null
} = {}) {
    if (!Number.isFinite(valueStageIndex)) return null;

    const safeValueStageIndex = Math.max(0, Math.floor(valueStageIndex));
    const safeFirstProjectionStageIndex = Number.isFinite(firstProjectionStageIndex)
        ? Math.max(0, Math.floor(firstProjectionStageIndex))
        : null;
    const projectionComponents = [];
    const seen = new Set();
    const pushComponent = (stageIndex, componentKey) => {
        const stateKey = buildMhsaProjectionComponentStateKey(stageIndex, componentKey);
        if (!stateKey || seen.has(stateKey)) return;
        seen.add(stateKey);
        projectionComponents.push(stateKey);
    };

    if (Number.isFinite(safeFirstProjectionStageIndex)) {
        pushComponent(safeFirstProjectionStageIndex, 'input');
    }

    MHSA_FULL_PROJECTION_STAGE_COMPONENT_KEYS.forEach((componentKey) => {
        pushComponent(safeValueStageIndex, componentKey);
    });

    return {
        projectionComponents,
        attentionBlocks: ['value-post'],
        connectors: ['v']
    };
}

export function resolveMhsaValueInputRowSceneFocus({
    valueStageIndex = null,
    firstProjectionStageIndex = null
} = {}) {
    return resolveMhsaProjectionInputRowSceneFocus({
        stageIndex: valueStageIndex,
        firstProjectionStageIndex
    });
}

function resolveMhsaTokenMatrixProjectionStageComponentKey(target, stageEl) {
    if (!(target instanceof Element) || !(stageEl instanceof Element)) return null;
    for (const spec of MHSA_TOKEN_MATRIX_PROJECTION_STAGE_COMPONENT_SPECS) {
        const componentEl = target.closest(spec.selector);
        if (componentEl instanceof Element && stageEl.contains(componentEl)) {
            return spec.key;
        }
    }
    return null;
}

export function resolveMhsaTokenMatrixProjectionStageTarget(target, {
    root = null,
    requirePinnableComponent = false
} = {}) {
    if (!(target instanceof Element)) return null;

    const stageEl = target.closest(MHSA_TOKEN_MATRIX_PROJECTION_STAGE_SELECTOR);
    if (!(stageEl instanceof Element)) return null;
    if (root instanceof Element && !root.contains(stageEl)) return null;

    if (requirePinnableComponent) {
        const componentEl = target.closest(MHSA_TOKEN_MATRIX_PINNABLE_PROJECTION_STAGE_COMPONENT_SELECTOR);
        if (!(componentEl instanceof Element) || !stageEl.contains(componentEl)) {
            return null;
        }
    }

    const stageIndex = Number(stageEl.dataset.mhsaProjectionStageIndex);
    if (!Number.isFinite(stageIndex)) return null;
    const componentKey = resolveMhsaTokenMatrixProjectionStageComponentKey(target, stageEl);

    return {
        stageIndex,
        focusKey: componentKey
    };
}
