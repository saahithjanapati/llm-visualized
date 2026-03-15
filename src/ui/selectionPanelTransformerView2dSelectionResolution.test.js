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
});
