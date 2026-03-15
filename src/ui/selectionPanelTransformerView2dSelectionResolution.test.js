import * as THREE from 'three';
import { beforeAll, describe, expect, it, vi } from 'vitest';

let SelectionPanel;

beforeAll(async () => {
    vi.stubGlobal('localStorage', {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {}
    });
    ({ SelectionPanel } = await import('./selectionPanel.js'));
});

function createPanelContext(scene = new THREE.Scene()) {
    const panel = Object.create(SelectionPanel.prototype);
    panel.engine = {
        scene,
        _layers: []
    };
    panel.pipeline = null;
    panel.activationSource = null;
    panel._lastSelection = null;
    return panel;
}

function createSceneNode(label = '', metadata = {}) {
    const node = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshBasicMaterial()
    );
    node.userData = {
        label,
        ...metadata
    };
    return node;
}

function createExpandedMlpVectorRef({
    prismCount = 12,
    vectorIndex = 0,
    activationData = null
} = {}) {
    const mesh = new THREE.InstancedMesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshBasicMaterial(),
        prismCount * 3
    );
    mesh.userData = {
        isVector: true,
        prismCount,
        instanceKind: 'batchedVector',
        label: 'MLP Expanded Segments'
    };

    const vectorRef = {
        isBatchedVectorRef: true,
        _index: vectorIndex,
        _batch: {
            prismCount,
            mesh
        },
        mesh,
        group: new THREE.Object3D(),
        instanceCount: prismCount,
        userData: {
            activationData: activationData || {}
        }
    };
    vectorRef.group.userData = {
        isVector: true
    };
    return {
        mesh,
        vectorRef
    };
}

describe('SelectionPanel 2D canvas selection resolution', () => {
    it('recovers live attention vectors by label when the scene node lacks stage metadata', () => {
        const scene = new THREE.Scene();
        const keyVectorNode = createSceneNode('Key Vector', {
            layerIndex: 6,
            headIndex: 4,
            tokenIndex: 3,
            tokenLabel: 'target'
        });
        scene.add(keyVectorNode);

        const panel = createPanelContext(scene);
        const selection = {
            label: 'Key Vector',
            info: {
                activationData: {
                    label: 'Key Vector',
                    stage: 'qkv.k',
                    layerIndex: 6,
                    headIndex: 4,
                    tokenIndex: 3,
                    tokenLabel: 'target'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.object).toBe(keyVectorNode);
        expect(resolved?.label).toBe('Key Vector');
        expect(resolved?.kind).toBe('vector');
    });

    it('recovers live layer norm parameter nodes by label when strict stage lookup misses', () => {
        const scene = new THREE.Scene();
        const layerNormScaleNode = createSceneNode('LayerNorm 1 Scale', {
            layerIndex: 2,
            layerNormKind: 'ln1'
        });
        scene.add(layerNormScaleNode);

        const panel = createPanelContext(scene);
        const selection = {
            label: 'LayerNorm 1 Scale',
            info: {
                layerNormKind: 'ln1',
                activationData: {
                    label: 'LayerNorm 1 Scale',
                    stage: 'ln1.param.scale',
                    layerIndex: 2,
                    layerNormKind: 'ln1'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.object).toBe(layerNormScaleNode);
        expect(resolved?.label).toBe('LayerNorm 1 Scale');
        expect(resolved?.kind).toBe('vector');
    });

    it('recovers the live expanded MLP-up mesh for 2D selections even when runtime segments are on activation stage', () => {
        const panel = createPanelContext();
        const { mesh, vectorRef } = createExpandedMlpVectorRef({
            prismCount: 12,
            vectorIndex: 1,
            activationData: {
                label: 'MLP Activation (post GELU)',
                stage: 'mlp.activation',
                layerIndex: 2,
                tokenIndex: 1,
                tokenLabel: 'target'
            }
        });
        panel.engine._layers = [
            null,
            null,
            {
                lanes: [{
                    tokenIndex: 1,
                    tokenLabel: 'target',
                    expandedVecSegments: [vectorRef]
                }]
            }
        ];

        const selection = {
            label: 'MLP Up Projection',
            info: {
                activationData: {
                    label: 'MLP Up Projection',
                    stage: 'mlp.up',
                    layerIndex: 2,
                    tokenIndex: 1,
                    tokenLabel: 'target'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.object).toBe(mesh);
        expect(resolved?.info?.vectorRef).toBe(vectorRef);
        expect(resolved?.label).toBe('MLP Up Projection');
        expect(resolved?.info?.activationData?.stage).toBe('mlp.up');
        expect(resolved?.kind).toBe('batchedVector');
        expect(resolved?.hit?.instanceId).toBe(12);
    });

    it('recovers the live expanded MLP activation mesh for 2D selections even when runtime segments still carry up-projection metadata', () => {
        const panel = createPanelContext();
        const { mesh, vectorRef } = createExpandedMlpVectorRef({
            prismCount: 12,
            activationData: {
                label: 'MLP Up Projection',
                stage: 'mlp.up',
                layerIndex: 4,
                tokenIndex: 0,
                tokenLabel: 'prompt'
            }
        });
        panel.engine._layers = [
            null,
            null,
            null,
            null,
            {
                lanes: [{
                    tokenIndex: 0,
                    tokenLabel: 'prompt',
                    expandedVecSegments: [vectorRef]
                }]
            }
        ];

        const selection = {
            label: 'MLP Activation (post GELU)',
            info: {
                activationData: {
                    label: 'MLP Activation (post GELU)',
                    stage: 'mlp.activation',
                    layerIndex: 4,
                    tokenIndex: 0,
                    tokenLabel: 'prompt'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.object).toBe(mesh);
        expect(resolved?.info?.vectorRef).toBe(vectorRef);
        expect(resolved?.label).toBe('MLP Activation (post GELU)');
        expect(resolved?.info?.activationData?.stage).toBe('mlp.activation');
        expect(resolved?.kind).toBe('batchedVector');
    });

    it('recovers the live ln2 residual vector for MLP detail source selections', () => {
        const panel = createPanelContext();
        const residualMesh = createSceneNode('Post-LayerNorm Residual Vector', {
            isVector: true,
            activationData: {
                label: 'Post-LayerNorm Residual Vector',
                stage: 'ln2.output',
                sourceStage: 'ln2.shift',
                layerIndex: 3,
                tokenIndex: 2,
                tokenLabel: 'source'
            }
        });
        const residualGroup = new THREE.Object3D();
        residualGroup.userData = {
            isVector: true,
            label: 'Post-LayerNorm Residual Vector'
        };
        residualGroup.add(residualMesh);
        const vectorRef = {
            isBatchedVectorRef: false,
            instanceCount: 12,
            rawData: new Array(12).fill(1),
            userData: {
                activationData: {
                    label: 'Post-LayerNorm Residual Vector',
                    stage: 'ln2.output',
                    sourceStage: 'ln2.shift',
                    layerIndex: 3,
                    tokenIndex: 2,
                    tokenLabel: 'source'
                }
            },
            mesh: residualMesh,
            group: residualGroup
        };
        panel.engine._layers = [
            null,
            null,
            null,
            {
                lanes: [{
                    tokenIndex: 2,
                    tokenLabel: 'source',
                    resultVecLN2: vectorRef
                }]
            }
        ];

        const selection = {
            label: 'Post-LayerNorm Residual Vector',
            info: {
                activationData: {
                    label: 'Post-LayerNorm Residual Vector',
                    stage: 'ln2.output',
                    sourceStage: 'ln2.shift',
                    layerIndex: 3,
                    tokenIndex: 2,
                    tokenLabel: 'source'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.object).toBe(vectorRef.group);
        expect(resolved?.info?.vectorRef).toBe(vectorRef);
        expect(resolved?.label).toBe('Post LayerNorm 2 Residual Vector');
        expect(resolved?.info?.activationData?.stage).toBe('ln2.output');
        expect(resolved?.info?.activationData?.sourceStage).toBe('ln2.shift');
        expect(resolved?.kind).toBe('vector');
    });

    it('recovers the live MLP-down vector for MLP detail output selections', () => {
        const panel = createPanelContext();
        const downMesh = createSceneNode('MLP Down Projection', {
            isVector: true,
            activationData: {
                label: 'MLP Down Projection',
                stage: 'mlp.down',
                layerIndex: 5,
                tokenIndex: 1,
                tokenLabel: 'output'
            }
        });
        const downGroup = new THREE.Object3D();
        downGroup.userData = {
            isVector: true,
            label: 'MLP Down Projection'
        };
        downGroup.add(downMesh);
        const vectorRef = {
            isBatchedVectorRef: false,
            instanceCount: 12,
            rawData: new Array(12).fill(2),
            userData: {
                activationData: {
                    label: 'MLP Down Projection',
                    stage: 'mlp.down',
                    layerIndex: 5,
                    tokenIndex: 1,
                    tokenLabel: 'output'
                }
            },
            mesh: downMesh,
            group: downGroup
        };
        panel.engine._layers = [
            null,
            null,
            null,
            null,
            null,
            {
                lanes: [{
                    tokenIndex: 1,
                    tokenLabel: 'output',
                    finalVecAfterMlp: vectorRef
                }]
            }
        ];

        const selection = {
            label: 'MLP Down Projection',
            info: {
                activationData: {
                    label: 'MLP Down Projection',
                    stage: 'mlp.down',
                    layerIndex: 5,
                    tokenIndex: 1,
                    tokenLabel: 'output'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.object).toBe(vectorRef.group);
        expect(resolved?.info?.vectorRef).toBe(vectorRef);
        expect(resolved?.label).toBe('MLP Down Projection');
        expect(resolved?.info?.activationData?.stage).toBe('mlp.down');
        expect(resolved?.kind).toBe('vector');
    });
});
