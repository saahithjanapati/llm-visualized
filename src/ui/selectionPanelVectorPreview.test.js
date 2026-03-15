import * as THREE from 'three';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import { FINAL_MLP_COLOR } from './selectionPanelConstants.js';
import {
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MHA_VALUE_SPECTRUM_COLOR
} from '../animations/LayerAnimationConstants.js';
import {
    MLP_DOWN_BIAS_TOOLTIP_LABEL,
    MLP_UP_BIAS_TOOLTIP_LABEL
} from '../utils/mlpLabels.js';

let buildVectorClonePreview;

beforeAll(async () => {
    vi.stubGlobal('localStorage', {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {}
    });
    ({ buildVectorClonePreview } = await import('./selectionPanel.js'));
});

function createSceneBackedVectorMesh({
    prismCount = 3,
    vectorColors = [0xFF0000, 0x00FF00, 0x0000FF]
} = {}) {
    const totalInstances = prismCount * vectorColors.length;
    const mesh = new THREE.InstancedMesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshBasicMaterial({ color: 0xFFFFFF }),
        totalInstances
    );
    mesh.userData = {
        isVector: true,
        prismCount,
        instanceKind: 'instanced'
    };
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(totalInstances * 3), 3);

    for (let instanceId = 0; instanceId < totalInstances; instanceId += 1) {
        const matrix = new THREE.Matrix4().makeTranslation(instanceId * 1.5, 0, 0);
        mesh.setMatrixAt(instanceId, matrix);
    }

    vectorColors.forEach((hex, vectorIndex) => {
        const color = new THREE.Color(hex);
        const baseInstanceId = vectorIndex * prismCount;
        for (let prismIndex = 0; prismIndex < prismCount; prismIndex += 1) {
            mesh.setColorAt(baseInstanceId + prismIndex, color);
        }
    });

    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    return mesh;
}

function createExpandedMlpBatch({
    prismCount = 12,
    vectorColors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00],
    segmentGap = 120
} = {}) {
    const segmentCount = vectorColors.length;
    const totalInstances = prismCount * segmentCount;
    const mesh = new THREE.InstancedMesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshBasicMaterial({ color: 0xFFFFFF }),
        totalInstances
    );
    mesh.userData = {
        isVector: true,
        prismCount,
        instanceKind: 'batchedVector',
        label: 'MLP Expanded Segments'
    };
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(totalInstances * 3), 3);

    const vectorRefs = vectorColors.map((hex, vectorIndex) => {
        const values = new Array(prismCount * 64).fill(vectorIndex + 1);
        return {
            isBatchedVectorRef: true,
            _index: vectorIndex,
            _batch: null,
            instanceCount: prismCount,
            rawData: values.slice(),
            userData: {
                activationData: {
                    label: 'MLP Up Projection',
                    stage: 'mlp.up',
                    segmentIndex: vectorIndex,
                    values
                }
            },
            currentKeyColors: [],
            group: new THREE.Object3D(),
            mesh
        };
    });
    const batch = {
        prismCount,
        vectorCount: segmentCount,
        mesh,
        _vectorRefs: vectorRefs
    };
    vectorRefs.forEach((vectorRef) => {
        vectorRef._batch = batch;
        vectorRef.group.userData = {
            isVector: true,
            label: 'MLP Up Projection'
        };
    });

    vectorColors.forEach((hex, vectorIndex) => {
        const color = new THREE.Color(hex);
        const baseInstanceId = vectorIndex * prismCount;
        const segmentOffsetX = vectorIndex * segmentGap;
        for (let prismIndex = 0; prismIndex < prismCount; prismIndex += 1) {
            const instanceId = baseInstanceId + prismIndex;
            const matrix = new THREE.Matrix4().makeTranslation(segmentOffsetX + prismIndex * 1.5, 0, 0);
            mesh.setMatrixAt(instanceId, matrix);
            mesh.setColorAt(instanceId, color);
        }
    });

    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    return {
        mesh,
        vectorRefs
    };
}

function findPreviewMesh(root = null) {
    if (!root) return null;
    let previewMesh = null;
    root.traverse((child) => {
        if (!previewMesh && child?.isInstancedMesh) {
            previewMesh = child;
        }
    });
    return previewMesh;
}

function countPreviewMeshes(root = null) {
    let count = 0;
    root?.traverse?.((child) => {
        if (child?.isInstancedMesh) count += 1;
    });
    return count;
}

function getInstanceX(mesh, instanceId) {
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    mesh.getMatrixAt(instanceId, matrix);
    matrix.decompose(position, quaternion, scale);
    return position.x;
}

function getGradientColor(mesh, instanceId = 0, attributeName = 'colorStart') {
    const attribute = mesh?.geometry?.getAttribute?.(attributeName);
    if (!attribute) return null;
    return new THREE.Color(
        attribute.getX(instanceId),
        attribute.getY(instanceId),
        attribute.getZ(instanceId)
    );
}

function getHueDistance(color, baseHex) {
    if (!color) return Number.POSITIVE_INFINITY;
    const baseHsl = { h: 0, s: 0, l: 0 };
    const sampleHsl = { h: 0, s: 0, l: 0 };
    new THREE.Color(baseHex).getHSL(baseHsl);
    color.getHSL(sampleHsl);
    const rawDelta = Math.abs(sampleHsl.h - baseHsl.h);
    return Math.min(rawDelta, 1 - rawDelta);
}

function buildAttentionBiasSelection(label, stage, layerIndex = 4, headIndex = 2) {
    return {
        label,
        kind: 'vector',
        info: {
            activationData: {
                label,
                stage,
                layerIndex,
                headIndex
            },
            layerIndex,
            headIndex
        }
    };
}

function buildQkvHeadSelection(label, stage, values, layerIndex = 4, headIndex = 2, tokenIndex = 1) {
    return {
        label,
        kind: 'vector',
        info: {
            activationData: {
                label,
                stage,
                layerIndex,
                headIndex,
                tokenIndex,
                values
            },
            layerIndex,
            headIndex,
            tokenIndex,
            values
        }
    };
}

function buildAttentionOutputSelection(
    label = 'Attention Output Vector',
    stage = 'attention.output_projection',
    values = new Array(12).fill(0).map((_, index) => Math.sin(index / 3)),
    layerIndex = 4,
    tokenIndex = 1
) {
    return {
        label,
        kind: 'vector',
        info: {
            activationData: {
                label,
                stage,
                layerIndex,
                tokenIndex,
                values
            },
            layerIndex,
            tokenIndex,
            values
        }
    };
}

describe('buildVectorClonePreview', () => {
    it('copies only the selected scene-backed vector slice when prismCount metadata is present', () => {
        const prismCount = 3;
        const sourceMesh = createSceneBackedVectorMesh({
            prismCount,
            vectorColors: [0xFF0000, 0x00FF00, 0x0000FF]
        });
        const selection = {
            label: 'Post-LayerNorm Residual Vector',
            kind: 'instanced',
            object: sourceMesh,
            hit: {
                object: sourceMesh,
                instanceId: prismCount
            },
            info: {
                activationData: {
                    stage: 'ln1.output',
                    sourceStage: 'ln1.shift'
                },
                layerIndex: 2,
                tokenIndex: 1
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(prismCount);

        const copiedColor = new THREE.Color();
        previewMesh.getColorAt(0, copiedColor);
        expect(copiedColor.r).toBeCloseTo(0, 5);
        expect(copiedColor.g).toBeCloseTo(1, 5);
        expect(copiedColor.b).toBeCloseTo(0, 5);

        preview.dispose?.();
        sourceMesh.geometry.dispose();
        sourceMesh.material.dispose();
    });

    it('renders MLP-up as one expanded preview vector with the full batch colors', () => {
        const prismCount = 12;
        const { mesh: sourceMesh, vectorRefs } = createExpandedMlpBatch({
            prismCount,
            vectorColors: [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00],
            segmentGap: 120
        });
        const circularVectorRef = vectorRefs[1];
        sourceMesh.userData.vectorRef = circularVectorRef;

        const selection = {
            label: 'MLP Up Projection',
            kind: 'instanced',
            object: sourceMesh,
            hit: {
                object: sourceMesh,
                instanceId: prismCount
            },
            info: {
                vectorIndex: 1,
                vectorRef: circularVectorRef,
                activationData: {
                    label: 'MLP Up Projection',
                    stage: 'mlp.up',
                    layerIndex: 2,
                    tokenIndex: 1,
                    segmentIndex: 1
                },
                layerIndex: 2,
                tokenIndex: 1,
                prismCount
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(prismCount * 4);

        const sourceSegmentGap = getInstanceX(sourceMesh, prismCount) - getInstanceX(sourceMesh, prismCount - 1);
        const previewSegmentGap = getInstanceX(previewMesh, prismCount) - getInstanceX(previewMesh, prismCount - 1);
        expect(sourceSegmentGap).toBeGreaterThan(50);
        expect(previewSegmentGap).toBeLessThan(20);

        const copiedColor = new THREE.Color();
        previewMesh.getColorAt(prismCount * 2, copiedColor);
        expect(copiedColor.r).toBeCloseTo(0, 5);
        expect(copiedColor.g).toBeCloseTo(0, 5);
        expect(copiedColor.b).toBeCloseTo(1, 5);

        previewMesh.getColorAt(prismCount * 3, copiedColor);
        expect(copiedColor.r).toBeCloseTo(1, 5);
        expect(copiedColor.g).toBeCloseTo(1, 5);
        expect(copiedColor.b).toBeCloseTo(0, 5);

        preview.dispose?.();
        sourceMesh.geometry.dispose();
        sourceMesh.material.dispose();
    });

    it('builds a single expanded MLP preview from activation values when no live vector mesh is available', () => {
        const values = new Array(3072).fill(0).map((_, index) => Math.sin(index / 32));
        const selection = {
            label: 'MLP Activation (post GELU)',
            kind: 'vector',
            info: {
                activationData: {
                    label: 'MLP Activation (post GELU)',
                    stage: 'mlp.activation',
                    values
                },
                values
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(48);
        const previewSegmentGap = getInstanceX(previewMesh, 12) - getInstanceX(previewMesh, 11);
        expect(previewSegmentGap).toBeLessThan(20);

        preview.dispose?.();
    });

    it('treats grouped 48-prism MLP values as an expanded vector instead of collapsing to one prism', () => {
        const groupedValues = new Array(48).fill(0).map((_, index) => (index % 12) / 12);
        const selection = {
            label: 'MLP Up Projection',
            kind: 'vector',
            info: {
                activationData: {
                    label: 'MLP Up Projection',
                    stage: 'mlp.up',
                    values: groupedValues
                },
                values: groupedValues
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(48);
        const previewSegmentGap = getInstanceX(previewMesh, 12) - getInstanceX(previewMesh, 11);
        expect(previewSegmentGap).toBeLessThan(20);

        preview.dispose?.();
    });

    it('renders data-backed ln2 residual selections as one vector instead of the three-lane placeholder', () => {
        const values = new Array(768).fill(0).map((_, index) => Math.cos(index / 24));
        const selection = {
            label: 'Post LayerNorm 2 Residual Vector',
            kind: 'vector',
            info: {
                activationData: {
                    label: 'Post LayerNorm 2 Residual Vector',
                    stage: 'ln2.output',
                    sourceStage: 'ln2.shift',
                    values
                },
                values
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();
        expect(countPreviewMeshes(preview?.object)).toBe(1);

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(12);

        preview.dispose?.();
    });

    it('builds exact ln2 residual previews from batched vectors with cyclic raycast metadata', () => {
        const prismCount = 12;
        const label = 'Post LayerNorm 2 Residual Vector';
        const { mesh: sourceMesh, vectorRefs } = createExpandedMlpBatch({
            prismCount,
            vectorColors: [0xFF0000, 0x00FF00, 0x0000FF],
            segmentGap: 120
        });
        const selectedRef = vectorRefs[1];
        const activationData = {
            label,
            stage: 'ln2.output',
            sourceStage: 'ln2.shift',
            layerIndex: 2,
            tokenIndex: 1
        };
        selectedRef.userData.activationData = activationData;
        selectedRef.group.userData = {
            isVector: true,
            label,
            layerIndex: 2
        };

        const cyclicEntry = {
            label,
            activationData,
            vectorRef: selectedRef,
            layerIndex: 2,
            tokenIndex: 1,
            category: 'residual'
        };
        sourceMesh.userData.instanceLabels = new Array(prismCount * vectorRefs.length).fill(label);
        sourceMesh.userData.instanceEntries = new Array(prismCount * vectorRefs.length).fill(cyclicEntry);

        const selection = {
            label,
            kind: 'vector',
            object: sourceMesh,
            hit: {
                object: sourceMesh,
                instanceId: prismCount
            },
            info: {
                vectorIndex: 1,
                vectorRef: selectedRef,
                activationData,
                layerIndex: 2,
                tokenIndex: 1,
                prismCount
            }
        };

        let preview = null;
        expect(() => {
            preview = buildVectorClonePreview(selection, selection.label);
        }).not.toThrow();
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(prismCount);

        const copiedColor = new THREE.Color();
        previewMesh.getColorAt(0, copiedColor);
        expect(copiedColor.r).toBeCloseTo(0, 5);
        expect(copiedColor.g).toBeCloseTo(1, 5);
        expect(copiedColor.b).toBeCloseTo(0, 5);

        preview.dispose?.();
        sourceMesh.geometry.dispose();
        sourceMesh.material.dispose();
    });

    it('renders data-backed MLP-down selections as one vector instead of the three-lane placeholder', () => {
        const values = new Array(768).fill(0).map((_, index) => Math.sin(index / 20));
        const selection = {
            label: 'MLP Down Projection',
            kind: 'vector',
            info: {
                activationData: {
                    label: 'MLP Down Projection',
                    stage: 'mlp.down',
                    values
                },
                values
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();
        expect(countPreviewMeshes(preview?.object)).toBe(1);

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(12);

        preview.dispose?.();
    });

    it('renders b_up previews with 48 MLP-colored prisms from the saved bias samples', () => {
        const selection = {
            label: MLP_UP_BIAS_TOOLTIP_LABEL,
            kind: 'vector',
            info: {
                activationData: {
                    label: MLP_UP_BIAS_TOOLTIP_LABEL,
                    stage: 'mlp.up.bias',
                    layerIndex: 4
                },
                layerIndex: 4
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(48);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, FINAL_MLP_COLOR)).toBeLessThan(0.12);
        expect(getHueDistance(endColor, FINAL_MLP_COLOR)).toBeLessThan(0.12);

        preview.dispose?.();
    });

    it('renders b_down previews with 12 MLP-colored prisms from the saved bias samples', () => {
        const selection = {
            label: MLP_DOWN_BIAS_TOOLTIP_LABEL,
            kind: 'vector',
            info: {
                activationData: {
                    label: MLP_DOWN_BIAS_TOOLTIP_LABEL,
                    stage: 'mlp.down.bias',
                    layerIndex: 4
                },
                layerIndex: 4
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(12);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, FINAL_MLP_COLOR)).toBeLessThan(0.12);
        expect(getHueDistance(endColor, FINAL_MLP_COLOR)).toBeLessThan(0.12);

        preview.dispose?.();
    });

    it('renders query bias previews as a single Q-colored head prism', () => {
        const selection = buildAttentionBiasSelection('Query Bias Vector', 'qkv.q.bias');

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(1);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, 0x276ebb)).toBeLessThan(0.14);
        expect(getHueDistance(endColor, 0x276ebb)).toBeLessThan(0.14);

        preview.dispose?.();
    });

    it('renders key bias previews as a single K-colored head prism', () => {
        const selection = buildAttentionBiasSelection('Key Bias Vector', 'qkv.k.bias');

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(1);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, 0x1e9f57)).toBeLessThan(0.14);
        expect(getHueDistance(endColor, 0x1e9f57)).toBeLessThan(0.14);

        preview.dispose?.();
    });

    it('renders value bias previews as a single V-colored head prism', () => {
        const selection = buildAttentionBiasSelection('Value Bias Vector', 'qkv.v.bias');

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(1);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, 0xc44d25)).toBeLessThan(0.14);
        expect(getHueDistance(endColor, 0xc44d25)).toBeLessThan(0.14);

        preview.dispose?.();
    });

    it('renders query vectors as a single Q-colored head prism in the detail-view preview', () => {
        const selection = buildQkvHeadSelection('Query Vector', 'qkv.q', new Array(64).fill(0));

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(1);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, 0x276ebb)).toBeLessThan(0.14);
        expect(getHueDistance(endColor, 0x276ebb)).toBeLessThan(0.14);

        preview.dispose?.();
    });

    it('renders key vectors as a single K-colored head prism in the detail-view preview', () => {
        const selection = buildQkvHeadSelection('Key Vector', 'qkv.k', new Array(64).fill(0));

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(1);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, 0x1e9f57)).toBeLessThan(0.14);
        expect(getHueDistance(endColor, 0x1e9f57)).toBeLessThan(0.14);

        preview.dispose?.();
    });

    it('renders value vectors as a single V-colored head prism in the detail-view preview', () => {
        const selection = buildQkvHeadSelection('Value Vector', 'qkv.v', new Array(64).fill(0));

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(1);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, 0xc44d25)).toBeLessThan(0.14);
        expect(getHueDistance(endColor, 0xc44d25)).toBeLessThan(0.14);

        preview.dispose?.();
    });

    it('renders output-projection bias previews with 12 purple prisms from the saved bias samples', () => {
        const selection = {
            label: 'Output Projection Bias Vector',
            kind: 'vector',
            info: {
                activationData: {
                    label: 'Output Projection Bias Vector',
                    stage: 'attention.output_projection.bias',
                    layerIndex: 4
                },
                layerIndex: 4
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(12);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, MHA_OUTPUT_PROJECTION_MATRIX_COLOR)).toBeLessThan(0.12);
        expect(getHueDistance(endColor, MHA_OUTPUT_PROJECTION_MATRIX_COLOR)).toBeLessThan(0.12);

        preview.dispose?.();
    });

    it('renders attention output vectors as a single residual-style preview vector', () => {
        const selection = buildAttentionOutputSelection();

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();
        expect(countPreviewMeshes(preview?.object)).toBe(1);

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(12);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(startColor).toBeTruthy();
        expect(endColor).toBeTruthy();

        preview.dispose?.();
    });

    it('renders attention weighted sums as a single value-colored head prism', () => {
        const values = new Array(64).fill(0).map((_, index) => Math.sin(index / 8));
        const selection = {
            label: 'Attention Weighted Sum',
            kind: 'vector',
            info: {
                activationData: {
                    label: 'Attention Weighted Sum',
                    stage: 'attention.weighted_sum',
                    layerIndex: 4,
                    headIndex: 2,
                    tokenIndex: 1,
                    values
                },
                layerIndex: 4,
                headIndex: 2,
                tokenIndex: 1,
                values
            }
        };

        const preview = buildVectorClonePreview(selection, selection.label);
        expect(preview).toBeTruthy();

        const previewMesh = findPreviewMesh(preview?.object);
        expect(previewMesh?.isInstancedMesh).toBe(true);
        expect(previewMesh?.count).toBe(1);

        const startColor = getGradientColor(previewMesh, 0, 'colorStart');
        const endColor = getGradientColor(previewMesh, 0, 'colorEnd');
        expect(getHueDistance(startColor, MHA_VALUE_SPECTRUM_COLOR)).toBeLessThan(0.14);
        expect(getHueDistance(endColor, MHA_VALUE_SPECTRUM_COLOR)).toBeLessThan(0.14);

        preview.dispose?.();
    });
});
