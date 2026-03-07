function hasExplicitQkvVectorLabel(lower = '') {
    return lower.includes('query vector')
        || lower.includes('key vector')
        || lower.includes('value vector');
}

export function normalizeRaycastLabel(label, info = null, object = null) {
    const raw = String(label || '');
    const lower = raw.toLowerCase();
    const stage = info?.activationData?.stage
        || object?.userData?.activationData?.stage
        || '';
    const stageLower = String(stage).toLowerCase();
    const explicitQkvLabel = hasExplicitQkvVectorLabel(lower);

    const isPostLayerNormResidual = !explicitQkvLabel && (
        lower.includes('post-layernorm residual')
        || lower.includes('post layernorm residual')
        || stageLower === 'ln1.shift'
        || stageLower === 'ln2.shift'
    );
    if (isPostLayerNormResidual) {
        return 'Post LayerNorm Residual Vector';
    }

    const isEmbeddingSum = lower.includes('embedding sum') || stageLower.startsWith('embedding.sum');
    const isResidualStage = lower.includes('incoming residual')
        || lower.includes('post-attention residual')
        || lower.includes('post attention residual')
        || lower.includes('post-mlp residual')
        || lower.includes('post mlp residual')
        || stageLower.startsWith('layer.incoming')
        || stageLower.includes('residual');
    if (isEmbeddingSum || isResidualStage) {
        return 'Residual Stream Vector';
    }
    return raw;
}

export function simplifyLayerNormParamHoverLabel(label, info = null, object = null) {
    const raw = String(label || '');
    const lower = raw.toLowerCase();
    const stage = info?.activationData?.stage
        || object?.userData?.activationData?.stage
        || '';
    const stageLower = String(stage).toLowerCase();

    const isLayerNormContext = lower.includes('layernorm')
        || lower.includes('layer norm')
        || lower.includes('ln1')
        || lower.includes('ln2')
        || lower.includes('final ln')
        || stageLower.includes('ln1.param.')
        || stageLower.includes('ln2.param.');
    if (!isLayerNormContext) return raw;

    const isScale = lower.includes('scale')
        || lower.includes('gamma')
        || lower.includes('γ')
        || stageLower.endsWith('.scale');
    if (isScale) return 'LayerNorm Scale';

    const isShift = lower.includes('shift')
        || lower.includes('beta')
        || lower.includes('β')
        || stageLower.endsWith('.shift');
    if (isShift) return 'LayerNorm Shift';

    return raw;
}

export function isCachedKvSelection(info = null, object = null) {
    const vectorRef = info?.vectorRef || null;
    if (vectorRef?.userData?.cachedKv === true || vectorRef?.userData?.kvCachePersistent === true) {
        return true;
    }
    if (vectorRef?.group?.userData?.cachedKv === true || vectorRef?.group?.userData?.kvCachePersistent === true) {
        return true;
    }
    if (vectorRef?.mesh?.userData?.cachedKv === true || vectorRef?.mesh?.userData?.kvCachePersistent === true) {
        return true;
    }
    let current = object || null;
    while (current) {
        const userData = current.userData || null;
        if (userData?.cachedKv === true || userData?.kvCachePersistent === true) {
            return true;
        }
        current = current.parent;
    }
    return false;
}
