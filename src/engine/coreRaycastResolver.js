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

    const visibleHits = intersects.filter((hit) => {
        const obj = hit?.object;
        return visibleFn(obj) && interactableFn(obj);
    });
    if (!visibleHits.length) return null;

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
    for (const hit of visibleHits) {
        const resolved = resolveKvProxyHit(hit);
        if (resolved) return resolved;
    }

    // Pass 1: Prefer detailed labels from merged K/V instanced meshes.
    for (const hit of visibleHits) {
        try {
            const obj = hit.object;
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
    }

    // Pass 1.25: Attention-sphere instanced mesh (per-instance activation data).
    for (const hit of visibleHits) {
        const obj = hit.object;
        if (!obj || !obj.isInstancedMesh || typeof hit.instanceId !== 'number') continue;
        if (!obj.userData || !obj.userData._attentionSphereInstanced) continue;
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
    }

    // Pass 1.4: Compact batched-vector metadata decoded from instanceId.
    for (const hit of visibleHits) {
        const obj = hit.object;
        if (!obj || !obj.isInstancedMesh || typeof hit.instanceId !== 'number') continue;
        const data = obj.userData || null;
        if (!data || data.instanceKind !== 'batchedVector') continue;
        if (data.raycastMetadataMode !== 'perVector') continue;
        const prismCount = Number.isFinite(data.prismCount) ? Math.max(1, Math.floor(data.prismCount)) : null;
        if (!prismCount) continue;
        const vectorEntries = Array.isArray(data.vectorEntries) ? data.vectorEntries : null;
        const vectorLabels = Array.isArray(data.vectorLabels) ? data.vectorLabels : null;
        if (!vectorEntries && !vectorLabels) continue;

        const vectorIndex = Math.floor(hit.instanceId / prismCount);
        if (!Number.isFinite(vectorIndex) || vectorIndex < 0) continue;
        const prismIndex = hit.instanceId % prismCount;
        const entry = vectorEntries && vectorIndex < vectorEntries.length ? vectorEntries[vectorIndex] : null;
        const label = (vectorLabels && vectorIndex < vectorLabels.length ? vectorLabels[vectorIndex] : null)
            || (entry && entry.label)
            || data.label
            || null;
        if (!label && !entry) continue;
        const info = entry && typeof entry === 'object'
            ? { ...entry, vectorIndex, prismIndex }
            : { vectorIndex, prismIndex };

        return {
            label: normalizeFn(label || 'Vector', info, obj),
            hit,
            info,
            kind: data.instanceKind || 'instanced'
        };
    }

    // Pass 1.5: Instance-specific labels for other instanced meshes.
    for (const hit of visibleHits) {
        const obj = hit.object;
        if (!obj || !obj.isInstancedMesh) continue;
        const labels = obj.userData?.instanceLabels;
        if (!Array.isArray(labels) || typeof hit.instanceId !== 'number') continue;
        const label = labels[hit.instanceId];
        if (!label) continue;
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
    }

    // Pass 1.75: Lightweight KV-cache raycast proxies.
    for (const hit of visibleHits) {
        const resolved = resolveKvProxyHit(hit);
        if (resolved) return resolved;
    }

    // Pass 2: Fallback – show first generic label.
    for (const hit of visibleHits) {
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
    }

    return null;
}

