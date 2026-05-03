import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { buildLayerNormPreview } from '../src/ui/selectionPanel.js';
import { resolveLayerNormParameterSummary } from '../src/ui/selectionPanelLayerNormPreviewUtils.js';

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

describe('selectionPanel layer norm preview', () => {
    it('uses the GPT-2 layer norm parameter width instead of the live display length', () => {
        const engine = {
            _layers: [
                {
                    index: 0,
                    _getBaseVectorLength: () => 1024
                }
            ]
        };

        const summary = resolveLayerNormParameterSummary({
            label: 'LayerNorm 1',
            info: {
                layerIndex: 0,
                layerNormKind: 'ln1'
            }
        }, engine);

        expect(summary.perParameterCount).toBe(768);
        expect(summary.totalParameterCount).toBe(1536);
        expect(summary.layerNormKind).toBe('ln1');
        expect(summary.layerIndex).toBe(0);
    });

    it('clones the selected layer norm solid without injecting synthetic param banks', () => {
        const sourceGroup = new THREE.Group();
        sourceGroup.userData.label = 'LayerNorm 2';

        const sourceGeometry = new THREE.TorusGeometry(32, 8, 12, 32);
        const sourceMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x68e3ff,
            transparent: true,
            opacity: 0.42
        });
        const sourceMesh = new THREE.InstancedMesh(sourceGeometry, sourceMaterial, 2);
        sourceMesh.userData.sourceMeshId = 'layer-norm-solid';

        const firstMatrix = new THREE.Matrix4().makeTranslation(0, 0, -18);
        const secondMatrix = new THREE.Matrix4().makeTranslation(0, 0, 18);
        sourceMesh.setMatrixAt(0, firstMatrix);
        sourceMesh.setMatrixAt(1, secondMatrix);
        sourceMesh.instanceMatrix.needsUpdate = true;
        sourceGroup.add(sourceMesh);

        const preview = buildLayerNormPreview('LayerNorm 2', {
            label: 'LayerNorm 2',
            object: sourceGroup,
            hit: { object: sourceMesh }
        });

        expect(preview?.object).toBeTruthy();
        expect(preview.object).not.toBe(sourceGroup);
        expect(preview.object.userData?.label).toBe('LayerNorm 2');

        const scaleGroups = collectLabelledGroups(preview.object, 'LayerNorm 2 Scale');
        const shiftGroups = collectLabelledGroups(preview.object, 'LayerNorm 2 Shift');
        expect(scaleGroups).toHaveLength(0);
        expect(shiftGroups).toHaveLength(0);

        let clonedMesh = null;
        preview.object.traverse((child) => {
            if (child?.isInstancedMesh && child.userData?.sourceMeshId === 'layer-norm-solid') {
                clonedMesh = child;
            }
        });

        expect(clonedMesh).toBeTruthy();
        expect(clonedMesh).not.toBe(sourceMesh);
        expect(clonedMesh.geometry).not.toBe(sourceGeometry);
        expect(clonedMesh.material).not.toBe(sourceMaterial);
        expect(clonedMesh.material.transparent).toBe(sourceMaterial.transparent);
        expect(clonedMesh.material.opacity).toBe(sourceMaterial.opacity);
        expect(clonedMesh.count).toBe(2);

        const clonedMatrix = new THREE.Matrix4();
        clonedMesh.getMatrixAt(0, clonedMatrix);
        expect(Array.from(clonedMatrix.elements)).toEqual(Array.from(firstMatrix.elements));
        clonedMesh.getMatrixAt(1, clonedMatrix);
        expect(Array.from(clonedMatrix.elements)).toEqual(Array.from(secondMatrix.elements));

        preview.dispose();
        sourceGeometry.dispose();
        sourceMaterial.dispose();
    });
});
