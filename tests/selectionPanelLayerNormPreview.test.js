import { describe, expect, it } from 'vitest';
import { buildLayerNormPreview } from '../src/ui/selectionPanel.js';

function makeLayer(index, {
    layoutCount = 1,
    activeLaneLayoutIndices = [0],
    baseVectorLength = 768
} = {}) {
    return {
        index,
        _laneLayoutCount: layoutCount,
        _activeLaneLayoutIndices: activeLaneLayoutIndices.slice(),
        _baseVectorLength: baseVectorLength,
        _getLaneLayoutCount() {
            return this._laneLayoutCount;
        },
        _getActiveLaneLayoutIndices() {
            return this._activeLaneLayoutIndices.slice();
        },
        _getBaseVectorLength() {
            return this._baseVectorLength;
        }
    };
}

function collectLabelledGroups(root, label) {
    const groups = [];
    root.traverse((child) => {
        if (child?.type !== 'Group') return;
        if (child.userData?.label === label) {
            groups.push(child);
        }
    });
    return groups;
}

function collectVectorMeshes(root) {
    const meshes = [];
    root.traverse((child) => {
        if (child?.isInstancedMesh && child.userData?.isVector) {
            meshes.push(child);
        }
    });
    return meshes;
}

describe('selectionPanel layer norm preview', () => {
    it('builds an opaque composite preview with blue active scale and shift vectors', () => {
        const layer = makeLayer(4, {
            layoutCount: 4,
            activeLaneLayoutIndices: [0, 3]
        });
        const preview = buildLayerNormPreview(
            'LayerNorm 2',
            {
                label: 'LayerNorm 2',
                info: {
                    layerIndex: 4,
                    layerNormKind: 'ln2'
                }
            },
            {
                _layers: [null, null, null, null, layer]
            }
        );

        expect(preview?.object).toBeTruthy();

        const previewObject = preview.object;
        const scaleGroups = collectLabelledGroups(previewObject, 'LayerNorm 2 Scale');
        const shiftGroups = collectLabelledGroups(previewObject, 'LayerNorm 2 Shift');
        const vectorMeshes = collectVectorMeshes(previewObject);

        expect(scaleGroups).toHaveLength(2);
        expect(shiftGroups).toHaveLength(2);
        expect(new Set(scaleGroups.map((group) => group.position.z)).size).toBe(2);
        expect(new Set(shiftGroups.map((group) => group.position.z)).size).toBe(2);
        expect(vectorMeshes).toHaveLength(4);

        const materials = [];
        previewObject.traverse((child) => {
            if (!child?.material) return;
            const mats = Array.isArray(child.material) ? child.material : [child.material];
            mats.forEach((mat) => {
                if (mat) materials.push(mat);
            });
        });
        expect(materials.length).toBeGreaterThan(0);
        materials.forEach((material) => {
            expect(material.transparent).toBe(false);
            expect(material.opacity).toBe(1);
        });

        const firstVectorMesh = vectorMeshes[0];
        expect(firstVectorMesh.instanceColor?.array).toBeTruthy();
        const colors = firstVectorMesh.instanceColor.array;
        let hasBlueActiveColor = false;
        for (let i = 0; i < colors.length; i += 3) {
            const r = colors[i];
            const g = colors[i + 1];
            const b = colors[i + 2];
            if (Math.abs(r - g) < 1e-4 && Math.abs(g - b) < 1e-4) continue;
            if (b > r) {
                hasBlueActiveColor = true;
                break;
            }
        }
        expect(hasBlueActiveColor).toBe(true);

        preview.dispose();
    });
});
