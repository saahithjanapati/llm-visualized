import { describe, expect, it } from 'vitest';
import { resolveRaycastLabel } from '../src/engine/coreRaycastResolver.js';

function makeObject(userData = {}, parent = null) {
    return { userData, parent };
}

describe('coreRaycastResolver', () => {
    it('returns null for empty hits', () => {
        const resolved = resolveRaycastLabel([], {});
        expect(resolved).toBe(null);
    });

    it('prefers kv proxy hits and normalizes label via callback', () => {
        const parent = makeObject({ label: 'Carrier' }, null);
        const proxy = makeObject(
            {
                kvRaycastProxy: true,
                vectorCategory: 'V',
                headIndex: 2,
                layerIndex: 1,
                cachedKv: true
            },
            parent
        );
        const hit = { object: proxy };
        const resolved = resolveRaycastLabel([hit], {
            isObjectVisible: () => true,
            isObjectInteractable: () => true,
            layers: [],
            normalizeLabel: (label) => `normalized:${label}`,
            isCachedKvSelection: () => false
        });

        expect(resolved).not.toBeNull();
        expect(resolved.kind).toBe('mergedKV');
        expect(resolved.label).toBe('normalized:Cached Value Vector');
        expect(resolved.object).toBe(parent);
        expect(resolved.info?.category).toBe('V');
    });

    it('falls back to first generic user label', () => {
        const labeled = makeObject({ label: 'Some Label' }, null);
        const hit = { object: labeled };
        const resolved = resolveRaycastLabel([hit], {
            isObjectVisible: () => true,
            isObjectInteractable: () => true,
            layers: [],
            normalizeLabel: (label) => label,
            isCachedKvSelection: () => false
        });
        expect(resolved?.kind).toBe('label');
        expect(resolved?.label).toBe('Some Label');
    });
});

