import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
    copyInstancedVectorSliceToPreview,
    shouldSkipLiveVectorTransformCopy,
    isInstancedVectorSliceInMotion,
} from '../src/ui/selectionPanelVectorCloneUtils.js';
import { HIDE_INSTANCE_Y_OFFSET } from '../src/utils/constants.js';

function makeAttrArray(length) {
    return {
        array: new Float32Array(length),
        needsUpdate: false
    };
}

function makeSourceMesh({ count = 3, yByIndex = null } = {}) {
    const colorStart = makeAttrArray(count * 3);
    const colorEnd = makeAttrArray(count * 3);
    const instanceColor = { array: new Float32Array(count * 3), needsUpdate: false };
    for (let i = 0; i < count * 3; i += 1) {
        colorStart.array[i] = i + 1;
        colorEnd.array[i] = (i + 1) * 2;
        instanceColor.array[i] = (i + 1) * 3;
    }
    return {
        isInstancedMesh: true,
        count,
        instanceMatrix: { count, needsUpdate: false },
        instanceColor,
        geometry: {
            getAttribute: (name) => {
                if (name === 'colorStart') return colorStart;
                if (name === 'colorEnd') return colorEnd;
                return null;
            }
        },
        getMatrixAt: (idx, target) => {
            const y = Array.isArray(yByIndex) ? yByIndex[idx] : idx + 10;
            target.identity();
            target.makeTranslation(idx + 1, y, idx + 100);
        }
    };
}

function makePreviewVec(instanceCount = 3) {
    const matrices = new Array(instanceCount).fill(null);
    const colorStart = makeAttrArray(instanceCount * 3);
    const colorEnd = makeAttrArray(instanceCount * 3);
    return {
        instanceCount,
        mesh: {
            setMatrixAt: (idx, matrix) => {
                matrices[idx] = matrix.clone();
            },
            instanceMatrix: { needsUpdate: false },
            instanceColor: { array: new Float32Array(instanceCount * 3), needsUpdate: false },
            geometry: {
                getAttribute: (name) => {
                    if (name === 'colorStart') return colorStart;
                    if (name === 'colorEnd') return colorEnd;
                    return null;
                }
            }
        },
        _matrices: matrices
    };
}

describe('selectionPanelVectorCloneUtils', () => {
    it('copies instanced matrix/color slices into preview vectors', () => {
        const source = makeSourceMesh({ count: 4 });
        const previewVec = makePreviewVec(3);
        const copied = copyInstancedVectorSliceToPreview(previewVec, source, 1, 3);

        expect(copied).toBe(true);
        expect(previewVec.mesh.instanceMatrix.needsUpdate).toBe(true);
        expect(previewVec.mesh.instanceColor.needsUpdate).toBe(true);
        expect(previewVec.mesh.geometry.getAttribute('colorStart').needsUpdate).toBe(true);
        expect(previewVec.mesh.geometry.getAttribute('colorEnd').needsUpdate).toBe(true);

        const outPos = new THREE.Vector3();
        previewVec._matrices[0].decompose(outPos, new THREE.Quaternion(), new THREE.Vector3());
        expect(outPos.x).toBeCloseTo(2);
    });

    it('detects motion when visible prisms have different Y positions', () => {
        const stable = makeSourceMesh({ count: 3, yByIndex: [10, 10, 10] });
        const moving = makeSourceMesh({ count: 3, yByIndex: [10, 11, 10] });
        expect(isInstancedVectorSliceInMotion(stable, 0, 3)).toBe(false);
        expect(isInstancedVectorSliceInMotion(moving, 0, 3)).toBe(true);
    });

    it('treats fully hidden vector slices as invalid live preview sources', () => {
        const hidden = makeSourceMesh({
            count: 3,
            yByIndex: [
                HIDE_INSTANCE_Y_OFFSET,
                HIDE_INSTANCE_Y_OFFSET,
                HIDE_INSTANCE_Y_OFFSET
            ]
        });
        const vectorRef = { mesh: hidden, instanceCount: 3 };

        expect(isInstancedVectorSliceInMotion(hidden, 0, 3)).toBe(true);
        expect(shouldSkipLiveVectorTransformCopy(null, vectorRef, null, 3)).toBe(true);
        expect(shouldSkipLiveVectorTransformCopy(null, vectorRef, null, 3, { forceLiveCopy: true })).toBe(true);
    });

    it('still skips fully hidden qkv-processed vectors', () => {
        const hidden = makeSourceMesh({
            count: 3,
            yByIndex: [
                HIDE_INSTANCE_Y_OFFSET,
                HIDE_INSTANCE_Y_OFFSET,
                HIDE_INSTANCE_Y_OFFSET
            ]
        });
        const vectorRef = {
            mesh: hidden,
            instanceCount: 3,
            userData: { qkvProcessed: true }
        };

        expect(shouldSkipLiveVectorTransformCopy(null, vectorRef, null, 3)).toBe(true);
    });

    it('skips live transform copy when vector slices are in motion', () => {
        const moving = makeSourceMesh({ count: 3, yByIndex: [4, 5, 4] });
        const vectorRef = { mesh: moving, instanceCount: 3 };
        expect(shouldSkipLiveVectorTransformCopy(null, vectorRef, null, 3)).toBe(true);
        expect(shouldSkipLiveVectorTransformCopy(null, vectorRef, null, 3, { forceLiveCopy: true })).toBe(false);
    });

    it('inspects the hinted shared-mesh slice instead of always using offset zero', () => {
        const source = makeSourceMesh({
            count: 6,
            yByIndex: [10, 10, 10, HIDE_INSTANCE_Y_OFFSET, HIDE_INSTANCE_Y_OFFSET, HIDE_INSTANCE_Y_OFFSET]
        });
        const vectorRef = {
            mesh: source,
            instanceCount: 3
        };
        const selectionInfo = {
            info: {
                vectorIndex: 1
            },
            hit: {
                object: source,
                instanceId: 4
            }
        };

        expect(shouldSkipLiveVectorTransformCopy(selectionInfo, vectorRef, null, 3)).toBe(true);
        expect(shouldSkipLiveVectorTransformCopy(null, vectorRef, null, 3)).toBe(false);
    });
});
