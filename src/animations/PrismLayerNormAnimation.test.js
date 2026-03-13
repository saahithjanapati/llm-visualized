import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { PrismLayerNormAnimation } from './PrismLayerNormAnimation.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';

function expectColorBufferToMatch(actual, expected) {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; i++) {
        expect(actual[i]).toBeCloseTo(expected[i], 6);
    }
}

describe('PrismLayerNormAnimation', () => {
    it('reveals deferred normalized vectors with the shared prism gradient state', () => {
        const initialData = [-1.5, -0.5, 0.5, 1.5];
        const normalizedData = [0.05, 0.35, 0.7, 1];
        const vec = new VectorVisualizationInstancedPrism(
            initialData,
            new THREE.Vector3(),
            4,
            initialData.length
        );
        const expectedState = vec.previewColorStateFromData(
            normalizedData,
            normalizedData.length,
            null,
            normalizedData
        );
        const anim = new PrismLayerNormAnimation(vec, {
            unitDelay: 1,
            unitDuration: 1
        });

        anim.start(normalizedData, {
            deferDataUpdate: true,
            sourceAlreadyNormalized: true
        });
        anim.update(0.05);
        anim.update(0.05);
        anim.update(0.05);

        const colorStartAttr = vec.mesh.geometry.getAttribute('colorStart');
        const colorEndAttr = vec.mesh.geometry.getAttribute('colorEnd');

        expect(anim.isAnimating).toBe(false);
        expect(expectedState).toBeTruthy();
        expectColorBufferToMatch(vec.mesh.instanceColor.array, expectedState.instanceColors);
        expectColorBufferToMatch(colorStartAttr.array, expectedState.colorStart);
        expectColorBufferToMatch(colorEndAttr.array, expectedState.colorEnd);
    });
});
