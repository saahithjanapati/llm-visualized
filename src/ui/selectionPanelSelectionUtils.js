import {
    expandLayerNormLabel,
    formatLayerNormParamLabel,
    isPostLayerNormResidualSelection,
    resolveLayerNormKind,
    resolvePostLayerNormResidualLabel,
    resolveLayerNormParamSpec
} from '../utils/layerNormLabels.js';

export function getActivationDataFromSelection(selectionInfo) {
    return selectionInfo?.info?.activationData
        || selectionInfo?.object?.userData?.activationData
        || selectionInfo?.hit?.object?.userData?.activationData
        || null;
}

function hasExplicitQkvVectorLabel(lower = '') {
    return lower.includes('query vector')
        || lower.includes('key vector')
        || lower.includes('value vector');
}

export function isKvCacheVectorSelection(selectionInfo) {
    const vectorRef = selectionInfo?.info?.vectorRef;
    if (vectorRef?.userData?.kvCachePersistent === true || vectorRef?.userData?.cachedKv === true) {
        return true;
    }
    const candidates = [selectionInfo?.object, selectionInfo?.hit?.object];
    for (const obj of candidates) {
        let current = obj;
        while (current && !current.isScene) {
            const userData = current.userData || null;
            if (
                userData?.kvCachePersistent === true
                || userData?.cachedKv === true
                || userData?.kvRaycastProxy === true
            ) {
                return true;
            }
            current = current.parent;
        }
    }
    return false;
}

export function normalizeSelectionLabel(label, selectionInfo = null) {
    const raw = String(label || '');
    const lower = raw.toLowerCase();
    const activation = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activation?.stage || '').toLowerCase();
    const explicitQkvLabel = hasExplicitQkvVectorLabel(lower);

    const isPostLayerNormResidual = !explicitQkvLabel && isPostLayerNormResidualSelection({
        label: raw,
        stage: activation?.stage || ''
    });
    if (isPostLayerNormResidual) {
        return resolvePostLayerNormResidualLabel({
            label: raw,
            stage: activation?.stage || ''
        });
    }

    const isEmbeddingSum = lower.includes('embedding sum') || stageLower.startsWith('embedding.sum');
    const isResidualStreamStage = lower.includes('incoming residual')
        || lower.includes('post-attention residual')
        || lower.includes('post attention residual')
        || lower.includes('post-mlp residual')
        || lower.includes('post mlp residual')
        || stageLower.startsWith('layer.incoming')
        || stageLower.includes('residual');

    if (isEmbeddingSum || isResidualStreamStage) {
        return 'Residual Stream Vector';
    }

    const cachedKv = isKvCacheVectorSelection(selectionInfo);
    if (cachedKv) {
        if (lower.includes('value vector')) return 'Cached Value Vector';
        if (lower.includes('key vector')) return 'Cached Key Vector';
        const category = String(selectionInfo?.info?.category || '').toUpperCase();
        if (category === 'V') return 'Cached Value Vector';
        if (category === 'K') return 'Cached Key Vector';
    }
    return raw;
}

export function simplifyLayerNormParamDisplayLabel(label, selectionInfo = null) {
    const raw = String(label || '');
    const lower = raw.toLowerCase();
    const stageLower = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    const activationStage = getActivationDataFromSelection(selectionInfo)?.stage || '';
    if (isPostLayerNormResidualSelection({ label: raw, stage: activationStage })) {
        return resolvePostLayerNormResidualLabel({
            label: raw,
            stage: activationStage,
            explicitKind: findUserDataString(selectionInfo, 'layerNormKind')
        });
    }
    const explicitKind = findUserDataString(selectionInfo, 'layerNormKind');
    const paramSpec = resolveLayerNormParamSpec({
        label: raw,
        stage: stageLower,
        explicitKind
    });
    const layerNormKind = resolveLayerNormKind({
        label: raw,
        stage: stageLower,
        explicitKind
    });

    if (lower.startsWith('mlp up projection')) return 'MLP Up Projection';
    if (lower.startsWith('mlp down projection')) return 'MLP Down Projection';

    if (paramSpec) {
        return formatLayerNormParamLabel(paramSpec.layerNormKind, paramSpec.param);
    }

    const isLayerNormContext = lower.includes('layernorm')
        || lower.includes('layer norm')
        || lower.includes('ln1')
        || lower.includes('ln2')
        || lower.includes('final ln')
        || stageLower.includes('ln1.param.')
        || stageLower.includes('ln2.param.')
        || stageLower.includes('final_ln.param.');
    if (!isLayerNormContext) return raw;

    const isScale = lower.includes('scale')
        || lower.includes('gamma')
        || lower.includes('γ')
        || stageLower.endsWith('.scale');
    if (isScale) return formatLayerNormParamLabel(layerNormKind, 'scale');

    const isShift = lower.includes('shift')
        || lower.includes('beta')
        || lower.includes('β')
        || stageLower.endsWith('.shift');
    if (isShift) return formatLayerNormParamLabel(layerNormKind, 'shift');

    return expandLayerNormLabel(raw, layerNormKind);
}

export function findUserDataNumber(selectionInfo, key) {
    const direct = selectionInfo?.info?.[key];
    if (Number.isFinite(direct)) return direct;
    const infoActivation = selectionInfo?.info?.activationData?.[key];
    if (Number.isFinite(infoActivation)) return infoActivation;
    const vectorRef = selectionInfo?.info?.vectorRef || null;
    const vectorRefCandidates = [
        vectorRef?.userData,
        vectorRef?.userData?.parentLane,
        vectorRef?.userData?.activationData,
        vectorRef?.group?.userData,
        vectorRef?.group?.userData?.parentLane,
        vectorRef?.group?.userData?.activationData,
        vectorRef?.mesh?.userData,
        vectorRef?.mesh?.userData?.parentLane,
        vectorRef?.mesh?.userData?.activationData
    ];
    for (const candidate of vectorRefCandidates) {
        if (candidate && Number.isFinite(candidate[key])) return candidate[key];
    }
    const candidates = [selectionInfo?.object, selectionInfo?.hit?.object];
    for (const obj of candidates) {
        let current = obj;
        while (current && !current.isScene) {
            const ud = current.userData;
            if (ud && Number.isFinite(ud[key])) return ud[key];
            if (ud?.activationData && Number.isFinite(ud.activationData[key])) return ud.activationData[key];
            if (ud?.parentLane && Number.isFinite(ud.parentLane[key])) return ud.parentLane[key];
            current = current.parent;
        }
    }
    return null;
}

export function findUserDataString(selectionInfo, key) {
    const direct = selectionInfo?.info?.[key];
    if (typeof direct === 'string') return direct;
    const infoActivation = selectionInfo?.info?.activationData?.[key];
    if (typeof infoActivation === 'string') return infoActivation;
    const vectorRef = selectionInfo?.info?.vectorRef || null;
    const vectorRefCandidates = [
        vectorRef?.userData,
        vectorRef?.userData?.parentLane,
        vectorRef?.userData?.activationData,
        vectorRef?.group?.userData,
        vectorRef?.group?.userData?.parentLane,
        vectorRef?.group?.userData?.activationData,
        vectorRef?.mesh?.userData,
        vectorRef?.mesh?.userData?.parentLane,
        vectorRef?.mesh?.userData?.activationData
    ];
    for (const candidate of vectorRefCandidates) {
        if (typeof candidate?.[key] === 'string') return candidate[key];
    }
    const candidates = [selectionInfo?.object, selectionInfo?.hit?.object];
    for (const obj of candidates) {
        let current = obj;
        while (current && !current.isScene) {
            const ud = current.userData;
            if (typeof ud?.[key] === 'string') return ud[key];
            if (typeof ud?.activationData?.[key] === 'string') return ud.activationData[key];
            if (typeof ud?.parentLane?.[key] === 'string') return ud.parentLane[key];
            current = current.parent;
        }
    }
    return null;
}

function isLogitEntryLike(entry) {
    return !!(
        entry
        && typeof entry === 'object'
        && (
            typeof entry.token === 'string'
            || Number.isFinite(entry.token_id)
            || Number.isFinite(entry.tokenId)
            || Number.isFinite(entry.prob)
            || Number.isFinite(entry.probability)
            || Number.isFinite(entry.logit)
        )
    );
}

export function resolveSelectionLogitEntry(selectionInfo) {
    const directInfo = selectionInfo?.info;
    if (isLogitEntryLike(directInfo)) {
        return directInfo;
    }
    if (isLogitEntryLike(directInfo?.logitEntry)) {
        return directInfo.logitEntry;
    }
    if (isLogitEntryLike(selectionInfo?.logitEntry)) {
        return selectionInfo.logitEntry;
    }

    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    const instanceId = selectionInfo?.hit?.instanceId;
    const entries = source?.userData?.instanceEntries;
    if (Array.isArray(entries) && Number.isFinite(instanceId) && instanceId >= 0 && instanceId < entries.length) {
        const entry = entries[instanceId];
        if (isLogitEntryLike(entry)) return entry;
    }

    const candidates = [selectionInfo?.object, selectionInfo?.hit?.object];
    for (const obj of candidates) {
        let current = obj;
        while (current && !current.isScene) {
            if (isLogitEntryLike(current.userData?.logitEntry)) {
                return current.userData.logitEntry;
            }
            current = current.parent;
        }
    }

    return null;
}

export function resolveAttentionModeFromSelection(selectionInfo) {
    const stage = getActivationDataFromSelection(selectionInfo)?.stage;
    if (stage === 'attention.post') return 'post';
    if (stage === 'attention.pre') return 'pre';
    if (stage === 'attention.mask') return 'pre';
    return null;
}

export function buildAttentionScoreLabel(mode = 'pre') {
    return mode === 'post'
        ? 'Post-Softmax Attention Score'
        : 'Pre-Softmax Attention Score';
}

export function matchesAttentionScoreSelection(selectionInfo, {
    mode = null,
    layerIndex = null,
    headIndex = null,
    tokenIndex = null,
    keyTokenIndex = null
} = {}) {
    if (!isAttentionScoreSelection(selectionInfo?.label, selectionInfo)) return false;

    const safeMode = mode === 'post' ? 'post' : (mode === 'pre' ? 'pre' : null);
    const stageLower = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    if (safeMode && stageLower !== `attention.${safeMode}`) return false;

    const matchNumber = (key, expected) => {
        if (!Number.isFinite(expected)) return true;
        const actual = findUserDataNumber(selectionInfo, key);
        return Number.isFinite(actual) && Math.floor(actual) === Math.floor(expected);
    };

    if (!matchNumber('layerIndex', layerIndex)) return false;
    if (!matchNumber('headIndex', headIndex)) return false;
    if (!matchNumber('tokenIndex', tokenIndex)) return false;
    if (!matchNumber('keyTokenIndex', keyTokenIndex)) return false;

    return true;
}

export function isValueSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (selectionInfo?.info?.category === 'V') return true;
    const stage = getActivationDataFromSelection(selectionInfo)?.stage;
    if (typeof stage === 'string' && stage.toLowerCase().startsWith('qkv.v')) return true;
    if (lower.includes('value vector')) return true;
    if (lower.includes('value weight matrix')) return true;
    if (lower.includes('merged value vectors')) return true;
    return false;
}

export function isWeightedSumSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('weighted sum')) return true;
    if (selectionInfo?.info?.isWeightedSum === true) return true;
    const candidates = [selectionInfo?.object, selectionInfo?.hit?.object];
    for (const obj of candidates) {
        let current = obj;
        while (current && !current.isScene) {
            if (current.userData?.isWeightedSum === true) return true;
            current = current.parent;
        }
    }
    return false;
}

export function isAttentionScoreSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('attention score')) return true;
    if (lower.includes('causal mask') || lower.includes('attention mask')) return true;
    const stage = selectionInfo?.info?.activationData?.stage
        || selectionInfo?.object?.userData?.activationData?.stage
        || selectionInfo?.hit?.object?.userData?.activationData?.stage;
    const stageLower = typeof stage === 'string' ? stage.toLowerCase() : '';
    if (stageLower === 'attention.pre' || stageLower === 'attention.post' || stageLower === 'attention.mask') return true;
    const kindLower = String(selectionInfo?.kind || '').toLowerCase();
    if (kindLower === 'attentionsphere') return true;
    const obj = selectionInfo?.object || selectionInfo?.hit?.object;
    return !!(obj && obj.isMesh && obj.geometry && obj.geometry.type === 'SphereGeometry');
}

export function isSelfAttentionSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (isAttentionScoreSelection(label, selectionInfo)) return true;
    if (selectionInfo?.kind === 'mergedKV') return true;
    // Keep pre-QKV residual copies out of the attention score panel while they travel to Q/K/V.
    if (isPostLayerNormResidualSelection({
        label,
        stage: getActivationDataFromSelection(selectionInfo)?.stage || ''
    })) return false;
    if (lower.includes('query vector') || lower.includes('key vector') || lower.includes('value vector')) return true;
    if (lower.includes('query weight matrix') || lower.includes('key weight matrix') || lower.includes('value weight matrix')) return true;
    if (lower.includes('merged key vectors') || lower.includes('merged value vectors')) return true;
    const stage = getActivationDataFromSelection(selectionInfo)?.stage;
    if (stage && (stage.startsWith('attention.') || stage.startsWith('qkv.'))) return true;
    return false;
}

export function isLogitBarSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (lower === 'logit' || lower.startsWith('logit ') || lower.includes('top logit bars')) {
        return true;
    }
    const kindLower = String(selectionInfo?.kind || '').toLowerCase();
    if (kindLower === 'logitbar') return true;
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    const instanceKindLower = String(source?.userData?.instanceKind || '').toLowerCase();
    return instanceKindLower === 'logitbar';
}

export function inferQkvType(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (selectionInfo?.info?.category === 'V') return 'V';
    if (selectionInfo?.info?.category === 'Q') return 'Q';
    if (selectionInfo?.info?.category === 'K') return 'K';
    if (lower.includes('value')) return 'V';
    if (lower.includes('query')) return 'Q';
    if (lower.includes('key')) return 'K';
    if (selectionInfo?.kind === 'mergedKV') return 'K';
    return 'K';
}

export function isWeightMatrixLabel(label) {
    const lower = (label || '').toLowerCase();
    return lower.includes('weight matrix')
        || lower.includes('embedding')
        || lower.includes('output projection matrix');
}

export function isQkvMatrixLabel(label) {
    const lower = (label || '').toLowerCase();
    return lower.includes('query weight matrix')
        || lower.includes('key weight matrix')
        || lower.includes('value weight matrix');
}

export function isParameterSelection(label) {
    const lower = (label || '').toLowerCase();
    if (isWeightMatrixLabel(lower)) return true;
    if (lower.includes('layernorm') || lower.includes('layer norm')) {
        if (lower.includes('scale') || lower.includes('shift') || lower.includes('gamma') || lower.includes('beta')) {
            return true;
        }
    }
    return false;
}

export function isLayerNormSolidSelection(label) {
    const lower = (label || '').toLowerCase();
    if (!(lower.includes('layernorm') || lower.includes('layer norm'))) return false;
    if (lower.includes('scale') || lower.includes('shift') || lower.includes('gamma') || lower.includes('beta')) return false;
    if (lower.includes('normed') || lower.includes('normalized') || lower.includes('output')) return false;
    if (lower.includes('residual') || lower.includes('vector')) return false;
    if (lower.includes('param')) return false;
    return true;
}

export function isResidualVectorSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('residual')) return true;
    const activation = getActivationDataFromSelection(selectionInfo);
    const activationLabel = activation?.label;
    if (typeof activationLabel === 'string' && activationLabel.toLowerCase().includes('residual')) return true;
    const stage = activation?.stage;
    if (typeof stage === 'string') {
        const stageLower = stage.toLowerCase();
        if (stageLower.includes('residual')) return true;
        if (stageLower.startsWith('layer.incoming')) return true;
        if (stageLower.startsWith('embedding.')) return true;
    }
    const cat = selectionInfo?.info?.category;
    if (cat && String(cat).toLowerCase().includes('residual')) return true;
    const kind = selectionInfo?.kind;
    if (kind && String(kind).toLowerCase().includes('residual')) return true;
    return false;
}

export function isLayerNormLabel(label) {
    const lower = (label || '').toLowerCase();
    return lower.includes('layernorm') || lower.includes('layer norm');
}
