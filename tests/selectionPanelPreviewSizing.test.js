import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import {
    buildVectorClonePreview,
    resolveLayerNormParamPreviewInstanceCount
} from '../src/ui/selectionPanel.js';
import { PRISM_DIMENSIONS_PER_UNIT } from '../src/utils/constants.js';

describe('selectionPanel preview sizing', () => {
    it('uses the grouped vector ref count for layernorm parameter previews when available', () => {
        const selection = {
            info: {
                vectorRef: {
                    instanceCount: 12
                }
            }
        };

        expect(resolveLayerNormParamPreviewInstanceCount(selection, new Array(768).fill(0))).toBe(12);
    });

    it('groups raw layernorm parameter data into preview prisms when no vector ref is available', () => {
        const rawData = new Array(PRISM_DIMENSIONS_PER_UNIT * 2 + 1).fill(0);

        expect(resolveLayerNormParamPreviewInstanceCount(null, rawData)).toBe(3);
    });

    it('clones the selected slice from a shared instanced vector mesh', () => {
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const sourceMesh = new THREE.InstancedMesh(geometry, material, 6);
        const sourceMatrix = new THREE.Matrix4();
        for (let index = 0; index < 6; index += 1) {
            sourceMesh.setMatrixAt(index, new THREE.Matrix4().makeTranslation(index < 3 ? index : 100 + index, 0, 0));
        }
        sourceMesh.instanceMatrix.needsUpdate = true;

        const preview = buildVectorClonePreview({
            label: 'Query Vector',
            kind: 'vector',
            info: {
                vectorIndex: 1,
                vectorRef: {
                    mesh: sourceMesh,
                    instanceCount: 3,
                    userData: {
                        activationData: { stage: 'qkv.q' }
                    }
                },
                activationData: { stage: 'qkv.q' }
            },
            hit: {
                object: sourceMesh,
                instanceId: 4
            }
        }, 'Query Vector');

        let previewMesh = null;
        preview?.object?.traverse((child) => {
            if (previewMesh || !child?.isInstancedMesh) return;
            previewMesh = child;
        });

        expect(previewMesh).toBeTruthy();
        expect(previewMesh?.count).toBe(3);

        const previewFirstMatrix = new THREE.Matrix4();
        const previewLastMatrix = new THREE.Matrix4();
        const expectedFirstMatrix = new THREE.Matrix4();
        const expectedLastMatrix = new THREE.Matrix4();
        previewMesh.getMatrixAt(0, previewFirstMatrix);
        previewMesh.getMatrixAt(2, previewLastMatrix);
        sourceMesh.getMatrixAt(3, expectedFirstMatrix);
        sourceMesh.getMatrixAt(5, expectedLastMatrix);

        expect(Array.from(previewFirstMatrix.elements)).toEqual(Array.from(expectedFirstMatrix.elements));
        expect(Array.from(previewLastMatrix.elements)).toEqual(Array.from(expectedLastMatrix.elements));

        preview?.dispose?.();
        geometry.dispose();
        material.dispose();
    });
});
