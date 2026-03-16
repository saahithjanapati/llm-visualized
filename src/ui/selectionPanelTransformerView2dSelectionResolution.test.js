import * as THREE from 'three';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import { CaptureActivationSource } from '../data/CaptureActivationSource.js';

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

    it('keeps layer norm input residual selections on activation-source data when only a broad scene vector exists', () => {
        const scene = new THREE.Scene();
        const sharedResidualMesh = new THREE.InstancedMesh(
            new THREE.BoxGeometry(1, 1, 1),
            new THREE.MeshBasicMaterial(),
            36
        );
        sharedResidualMesh.userData = {
            isVector: true,
            label: 'Incoming Residual (Pre-LN1)',
            activationData: {
                label: 'Incoming Residual (Pre-LN1)',
                stage: 'layer.incoming',
                layerIndex: 1,
                tokenIndex: 2,
                tokenLabel: 'source'
            }
        };
        scene.add(sharedResidualMesh);

        const panel = createPanelContext(scene);
        panel.activationSource = {
            getBaseVectorLength: () => 3,
            getTokenId: () => 17,
            getLayerIncoming: () => [0.1, 0.2, 0.3]
        };

        const selection = {
            label: 'Residual Stream Vector',
            info: {
                activationData: {
                    label: 'Residual Stream Vector',
                    stage: 'layer.incoming',
                    sourceStage: 'layer.incoming',
                    layerIndex: 1,
                    tokenIndex: 2,
                    tokenLabel: 'source'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.label).toBe('Residual Stream Vector');
        expect(resolved?.object).toBeUndefined();
        expect(resolved?.hit).toBeUndefined();
        expect(resolved?.info?.activationData?.stage).toBe('layer.incoming');
        expect(resolved?.info?.activationData?.values).toEqual([0.1, 0.2, 0.3]);
        expect(resolved?.info?.tokenId).toBe(17);
    });

    it('keeps layer norm product-vector selections on activation-source data when only a broad scene mesh exists', () => {
        const scene = new THREE.Scene();
        const sharedScaledMesh = new THREE.InstancedMesh(
            new THREE.BoxGeometry(1, 1, 1),
            new THREE.MeshBasicMaterial(),
            36
        );
        sharedScaledMesh.userData = {
            isVector: true,
            label: 'LN1 Scaled - source',
            activationData: {
                label: 'LN1 Scaled - source',
                stage: 'ln1.scale',
                layerIndex: 1,
                tokenIndex: 2,
                tokenLabel: 'source'
            }
        };
        scene.add(sharedScaledMesh);

        const panel = createPanelContext(scene);
        panel.activationSource = {
            getBaseVectorLength: () => 3,
            getTokenId: () => 17,
            getLayerLn1: (_layerIndex, stage) => (
                stage === 'scale' ? [0.4, 0.5, 0.6] : null
            )
        };

        const selection = {
            label: 'LayerNorm 1 Product Vector',
            info: {
                activationData: {
                    label: 'LayerNorm 1 Product Vector',
                    stage: 'ln1.product',
                    sourceStage: 'ln1.scale',
                    layerIndex: 1,
                    tokenIndex: 2,
                    tokenLabel: 'source',
                    layerNormKind: 'ln1'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.label).toBe('LayerNorm 1 Product Vector');
        expect(resolved?.object).toBeUndefined();
        expect(resolved?.hit).toBeUndefined();
        expect(resolved?.info?.activationData?.stage).toBe('ln1.product');
        expect(resolved?.info?.activationData?.sourceStage).toBe('ln1.scale');
        expect(resolved?.info?.activationData?.values).toEqual([0.4, 0.5, 0.6]);
        expect(resolved?.info?.tokenId).toBe(17);
    });

    it('keeps layer norm normalized-vector selections on activation-source data when only a broad scene mesh exists', () => {
        const scene = new THREE.Scene();
        const sharedNormalizedMesh = new THREE.InstancedMesh(
            new THREE.BoxGeometry(1, 1, 1),
            new THREE.MeshBasicMaterial(),
            36
        );
        sharedNormalizedMesh.userData = {
            isVector: true,
            label: 'LN1 Normalized - source',
            activationData: {
                label: 'LN1 Normalized - source',
                stage: 'ln1.norm',
                layerIndex: 1,
                tokenIndex: 2,
                tokenLabel: 'source'
            }
        };
        scene.add(sharedNormalizedMesh);

        const panel = createPanelContext(scene);
        panel.activationSource = {
            getBaseVectorLength: () => 3,
            getTokenId: () => 17,
            getLayerLn1: (_layerIndex, stage) => (
                stage === 'norm' ? [0.15, 0.25, 0.35] : null
            )
        };

        const selection = {
            label: 'LayerNorm 1 Normalized Vector',
            info: {
                activationData: {
                    label: 'LayerNorm 1 Normalized Vector',
                    stage: 'ln1.norm',
                    sourceStage: 'ln1.norm',
                    layerIndex: 1,
                    tokenIndex: 2,
                    tokenLabel: 'source',
                    layerNormKind: 'ln1'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.label).toBe('Normalized Residual Stream Vector');
        expect(resolved?.object).toBeUndefined();
        expect(resolved?.hit).toBeUndefined();
        expect(resolved?.info?.activationData?.stage).toBe('ln1.norm');
        expect(resolved?.info?.activationData?.values).toEqual([0.15, 0.25, 0.35]);
        expect(resolved?.info?.tokenId).toBe(17);
    });

    it('keeps layer norm output selections on activation-source data when only a broad scene mesh exists', () => {
        const scene = new THREE.Scene();
        const sharedOutputMesh = new THREE.InstancedMesh(
            new THREE.BoxGeometry(1, 1, 1),
            new THREE.MeshBasicMaterial(),
            36
        );
        sharedOutputMesh.userData = {
            isVector: true,
            label: 'Post-LayerNorm Residual Vector',
            activationData: {
                label: 'Post-LayerNorm Residual Vector',
                stage: 'ln1.output',
                sourceStage: 'ln1.shift',
                layerIndex: 1,
                tokenIndex: 2,
                tokenLabel: 'source'
            }
        };
        scene.add(sharedOutputMesh);

        const panel = createPanelContext(scene);
        panel.activationSource = {
            getBaseVectorLength: () => 3,
            getTokenId: () => 17,
            getLayerLn1: (_layerIndex, stage) => (
                stage === 'shift' ? [0.7, 0.8, 0.9] : null
            )
        };

        const selection = {
            label: 'Post LayerNorm 1 Residual Vector',
            info: {
                activationData: {
                    label: 'Post LayerNorm 1 Residual Vector',
                    stage: 'ln1.output',
                    sourceStage: 'ln1.shift',
                    layerIndex: 1,
                    tokenIndex: 2,
                    tokenLabel: 'source',
                    layerNormKind: 'ln1'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.label).toBe('Post LayerNorm 1 Residual Vector');
        expect(resolved?.object).toBeUndefined();
        expect(resolved?.hit).toBeUndefined();
        expect(resolved?.info?.activationData?.stage).toBe('ln1.output');
        expect(resolved?.info?.activationData?.sourceStage).toBe('ln1.shift');
        expect(resolved?.info?.activationData?.values).toEqual([0.7, 0.8, 0.9]);
        expect(resolved?.info?.tokenId).toBe(17);
    });

    it('keeps layer norm parameter selections off broad scene meshes', () => {
        const scene = new THREE.Scene();
        const sharedScaleMesh = new THREE.InstancedMesh(
            new THREE.BoxGeometry(1, 1, 1),
            new THREE.MeshBasicMaterial(),
            36
        );
        sharedScaleMesh.userData = {
            isVector: true,
            label: 'LayerNorm 1 Scale',
            activationData: {
                label: 'LayerNorm 1 Scale',
                stage: 'ln1.param.scale',
                layerIndex: 1,
                layerNormKind: 'ln1'
            }
        };
        scene.add(sharedScaleMesh);

        const panel = createPanelContext(scene);
        const selection = {
            label: 'LayerNorm 1 Scale',
            info: {
                activationData: {
                    label: 'LayerNorm 1 Scale',
                    stage: 'ln1.param.scale',
                    layerIndex: 1,
                    layerNormKind: 'ln1'
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.label).toBe('LayerNorm 1 Scale');
        expect(resolved?.object).toBeUndefined();
        expect(resolved?.hit).toBeUndefined();
        expect(resolved?.info?.activationData?.stage).toBe('ln1.param.scale');
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

    it('keeps causal-mask canvas selections on the dedicated attention.mask path', () => {
        const panel = createPanelContext();
        const selection = {
            label: 'Causal Mask',
            info: {
                activationData: {
                    label: 'Causal Mask',
                    stage: 'attention.mask',
                    layerIndex: 2,
                    headIndex: 1,
                    tokenIndex: 0,
                    queryTokenIndex: 0,
                    keyTokenIndex: 1,
                    tokenLabel: 'Token A',
                    keyTokenLabel: 'Token B',
                    preScore: 0.25,
                    postScore: 0,
                    maskValue: Number.NEGATIVE_INFINITY,
                    isMasked: true
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.label).toBe('Causal Mask');
        expect(resolved?.kind).toBe('attentionSphere');
        expect(resolved?.info?.activationData?.stage).toBe('attention.mask');
        expect(resolved?.info?.activationData?.maskValue).toBe(Number.NEGATIVE_INFINITY);
        expect(resolved?.info?.activationData?.isMasked).toBe(true);
    });

    it('keeps masked post-softmax 2D canvas selections at zero when packed capture rows stop at the diagonal', () => {
        const panel = createPanelContext();
        panel.activationSource = new CaptureActivationSource({
            meta: {
                token_strings: ['A', 'B', 'C']
            },
            activations: {
                layers: [{
                    attention_scores: {
                        post: [{
                            v: [0.6, 0.25, 0.75, 1],
                            n: 3
                        }]
                    }
                }]
            }
        });

        const selection = {
            label: 'Post-Softmax Attention Score',
            info: {
                activationData: {
                    label: 'Post-Softmax Attention Score',
                    stage: 'attention.post',
                    layerIndex: 0,
                    headIndex: 0,
                    tokenIndex: 0,
                    queryTokenIndex: 0,
                    keyTokenIndex: 2,
                    tokenLabel: 'A',
                    keyTokenLabel: 'C',
                    preScore: 0.4,
                    postScore: 0,
                    isMasked: true,
                    maskValue: Number.NEGATIVE_INFINITY
                }
            }
        };

        const resolved = panel._resolveTransformerView2dCanvasSelection(selection);

        expect(resolved?.label).toBe('Post-Softmax Attention Score');
        expect(resolved?.kind).toBe('attentionSphere');
        expect(resolved?.info?.activationData?.stage).toBe('attention.post');
        expect(resolved?.info?.activationData?.tokenIndex).toBe(0);
        expect(resolved?.info?.activationData?.keyTokenIndex).toBe(2);
        expect(resolved?.info?.activationData?.postScore).toBe(0);
    });

    it('renders causal-mask legends as a discrete black-or-zero palette', () => {
        const panel = createPanelContext();
        panel._resolveLegendEdgeClampRatio = () => 0;
        panel._resolveActiveAttentionDisplayMode = () => 'mask';

        const gradient = panel._buildAttentionLegendGradient('mask');
        const blockedSample = panel._resolveLegendHoverSample('attention', 0.2);
        const allowedSample = panel._resolveLegendHoverSample('attention', 0.8);

        expect(gradient).toContain('#000000 0%');
        expect(gradient).toContain('#000000 50%');
        expect(gradient).toContain('50%');
        expect(blockedSample?.valueLabel).toBe('-∞');
        expect(blockedSample?.colorCss).toBe('#000000');
        expect(allowedSample?.valueLabel).toBe('0');
        expect(allowedSample?.colorCss).not.toBe('#000000');
    });
});
