/**
 * Resolve the best label/metadata candidate from a list of raycast intersections.
 * This keeps multi-pass hit decoding logic out of CoreEngine orchestration code.
 */
export function resolveRaycastLabel(intersects, {
    isObjectVisible,
    isObjectInteractable,
    layers,
    normalizeLabel,
    isCachedKvSelection
} = {}) {
    if (!intersects || !intersects.length) return null;
    const visibleFn = typeof isObjectVisible === 'function' ? isObjectVisible : () => true;
    const interactableFn = typeof isObjectInteractable === 'function' ? isObjectInteractable : () => true;
    const normalizeFn = typeof normalizeLabel === 'function'
        ? normalizeLabel
        : (label) => String(label || '');
    const cacheCheckFn = typeof isCachedKvSelection === 'function'
        ? isCachedKvSelection
        : () => false;
    const layerList = Array.isArray(layers) ? layers : [];

    let hasVisibleHit = false;
    const forEachVisibleHit = (handler) => {
        for (let i = 0; i < intersects.length; i += 1) {
            const hit = intersects[i];
            const obj = hit?.object;
            if (!visibleFn(obj) || !interactableFn(obj)) continue;
            hasVisibleHit = true;
            const resolved = handler(hit, obj);
            if (resolved) return resolved;
        }
        return null;
    };

    const resolveKvProxyHit = (hit) => {
        const obj = hit?.object;
        if (!obj || !obj.userData?.kvRaycastProxy) return null;
        const proxyData = obj.userData || {};
        const category = String(proxyData.vectorCategory || 'K').toUpperCase() === 'V' ? 'V' : 'K';
        const info = {
            category,
            headIndex: Number.isFinite(proxyData.headIndex) ? proxyData.headIndex : null,
            layerIndex: Number.isFinite(proxyData.layerIndex) ? proxyData.layerIndex : null,
            laneLayoutIndex: Number.isFinite(proxyData.laneLayoutIndex) ? proxyData.laneLayoutIndex : null,
            tokenIndex: Number.isFinite(proxyData.tokenIndex) ? proxyData.tokenIndex : null
        };
        const carrier = obj.parent || obj;
        const cached = proxyData.cachedKv === true || cacheCheckFn(info, carrier);
        const catText = category === 'V'
            ? (cached ? 'Cached Value Vector' : 'Value Vector')
            : (cached ? 'Cached Key Vector' : 'Key Vector');
        return {
            label: normalizeFn(catText, info, carrier),
            hit,
            info,
            object: carrier,
            kind: 'mergedKV'
        };
    };

    // Pass 0.9: Prefer explicit KV-cache proxy hits when available.
    const proxyHit = forEachVisibleHit((hit) => resolveKvProxyHit(hit));
    if (proxyHit) return proxyHit;

    // Pass 1: Prefer detailed labels from merged K/V instanced meshes.
    const mergedKvHit = forEachVisibleHit((hit, obj) => {
        try {
            if (obj && obj.isInstancedMesh) {
                for (const layer of layerList) {
                    if (!layer || !layer.mhsaAnimation) continue;
                    const mhsa = layer.mhsaAnimation;
                    if (typeof mhsa.decodeMergedKVIntersection === 'function') {
                        const info = mhsa.decodeMergedKVIntersection(hit);
                        if (info) {
                            const cached = cacheCheckFn(info, hit.object);
                            const catText = info.category === 'V'
                                ? (cached ? 'Cached Value Vector' : 'Value Vector')
                                : (cached ? 'Cached Key Vector' : 'Key Vector');
                            return {
                                label: normalizeFn(catText, info),
                                hit,
                                info,
                                kind: 'mergedKV'
                            };
                        }
                    }
                }
            }
        } catch (_) {
            // Non-fatal decode failure; continue with fallback passes.
        }
        return null;
    });
    if (mergedKvHit) return mergedKvHit;

    // Pass 1.25: Attention-sphere instanced mesh (per-instance activation data).
    const attentionSphereHit = forEachVisibleHit((hit, obj) => {
        if (!obj || !obj.isInstancedMesh || typeof hit.instanceId !== 'number') return null;
        if (!obj.userData || !obj.userData._attentionSphereInstanced) return null;
        const labels = obj.userData.instanceLabels;
        const entries = obj.userData.instanceEntries;
        const label = Array.isArray(labels) ? labels[hit.instanceId] : 'Attention Score';
        const info = Array.isArray(entries) ? entries[hit.instanceId] : null;
        if (label || info) {
            return {
                label: normalizeFn(label || 'Attention Score', info, obj),
                hit,
                info,
                kind: 'attentionSphere'
            };
        }
        return null;
    });
    if (attentionSphereHit) return attentionSphereHit;

    // Pass 1.4: Compact batched-vector metadata decoded from instanceId.
    const batchedVectorHit = forEachVisibleHit((hit, obj) => {
        if (!obj || !obj.isInstancedMesh || typeof hit.instanceId !== 'number') return null;
        const data = obj.userData || null;
        if (!data || data.instanceKind !== 'batchedVector') return null;
        if (data.raycastMetadataMode !== 'perVector') return null;
        const prismCount = Number.isFinite(data.prismCount) ? Math.max(1, Math.floor(data.prismCount)) : null;
        if (!prismCount) return null;
        const vectorEntries = Array.isArray(data.vectorEntries) ? data.vectorEntries : null;
        const vectorLabels = Array.isArray(data.vectorLabels) ? data.vectorLabels : null;
        if (!vectorEntries && !vectorLabels) return null;

        const vectorIndex = Math.floor(hit.instanceId / prismCount);
        if (!Number.isFinite(vectorIndex) || vectorIndex < 0) return null;
        const prismIndex = hit.instanceId % prismCount;
        const entry = vectorEntries && vectorIndex < vectorEntries.length ? vectorEntries[vectorIndex] : null;
        const label = (vectorLabels && vectorIndex < vectorLabels.length ? vectorLabels[vectorIndex] : null)
            || (entry && entry.label)
            || data.label
            || null;
        if (!label && !entry) return null;
        const info = entry && typeof entry === 'object'
            ? { ...entry, vectorIndex, prismIndex }
            : { vectorIndex, prismIndex };

        return {
            label: normalizeFn(label || 'Vector', info, obj),
            hit,
            info,
            kind: data.instanceKind || 'instanced'
        };
    });
    if (batchedVectorHit) return batchedVectorHit;

    // Pass 1.5: Instance-specific labels for other instanced meshes.
    const instancedLabelHit = forEachVisibleHit((hit, obj) => {
        if (!obj || !obj.isInstancedMesh) return null;
        const labels = obj.userData?.instanceLabels;
        if (!Array.isArray(labels) || typeof hit.instanceId !== 'number') return null;
        const label = labels[hit.instanceId];
        if (!label) return null;
        const entries = obj.userData?.instanceEntries;
        const entry = Array.isArray(entries) ? entries[hit.instanceId] : null;
        const info = entry && typeof entry === 'object'
            ? entry
            : (entry !== undefined && entry !== null ? { logitEntry: entry } : null);
        return {
            label: normalizeFn(label, info, obj),
            hit,
            info,
            kind: obj.userData?.instanceKind || 'instanced'
        };
    });
    if (instancedLabelHit) return instancedLabelHit;

    // Pass 1.75: Lightweight KV-cache raycast proxies.
    const proxyFallbackHit = forEachVisibleHit((hit) => resolveKvProxyHit(hit));
    if (proxyFallbackHit) return proxyFallbackHit;

    // Pass 2: Fallback – show first generic label.
    const fallbackLabelHit = forEachVisibleHit((hit) => {
        let obj = hit.object;
        while (obj) {
            if (obj.userData?.kvRaycastProxy) {
                obj = obj.parent;
                continue;
            }
            const lbl = obj.userData?.label || obj.name;
            if (lbl && lbl !== 'Weight Matrix') {
                return {
                    label: normalizeFn(lbl, null, obj),
                    hit,
                    object: obj,
                    kind: 'label'
                };
            }
            obj = obj.parent;
        }
        return null;
    });
    if (fallbackLabelHit) return fallbackLabelHit;

    if (!hasVisibleHit) return null;

    return null;
}
